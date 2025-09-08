import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import math
from tqdm.auto import tqdm
import time
from typing import Dict, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GemmaTrainer:
    """
    Kaggle-optimized Gemma trainer with memory efficiency and checkpointing
    """
    
    def __init__(self, model, tokenizer, config):
        """
        Initialize the trainer
        
        Args:
            model: Gemma model instance
            tokenizer: Tokenizer instance
            config: Training configuration object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Mixed precision training
        self.use_amp = hasattr(config, 'use_amp') and config.use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using automatic mixed precision (AMP)")
        
        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
    def _setup_optimizer(self):
        """Setup optimizer with proper weight decay"""
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': getattr(self.config, 'weight_decay', 0.01),
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=getattr(self.config, 'learning_rate', 1e-4),
            betas=getattr(self.config, 'betas', (0.9, 0.95)),
            eps=getattr(self.config, 'eps', 1e-8),
        )
        
        logger.info(f"Optimizer setup with LR: {self.config.learning_rate}")
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        warmup_steps = getattr(self.config, 'warmup_steps', 1000)
        max_steps = getattr(self.config, 'max_steps', 10000)
        
        # Linear warmup + cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=getattr(self.config, 'min_lr', 1e-6)
        )
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        logger.info(f"Scheduler setup with warmup: {warmup_steps}, max_steps: {max_steps}")
        return scheduler
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss value
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Prepare inputs for causal language modeling
        # Shift inputs and labels for next token prediction
        if labels.equal(input_ids):
            # If labels weren't provided separately, create them from input_ids
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]  # Shift left
            labels[:, -1] = -100  # Don't predict beyond sequence
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        # Forward pass with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = self._forward_pass(input_ids, attention_mask, labels)
        else:
            loss = self._forward_pass(input_ids, attention_mask, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def _forward_pass(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                     labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Note: This needs to be adapted based on the exact Gemma model interface
        """
        # The exact implementation depends on how the Gemma model is structured
        # This is a generic implementation that might need adjustment
        
        try:
            # Try to use the model's forward method if it supports training
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                # Extract logits from outputs
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            # Calculate cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Flatten for loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # Return a dummy loss to prevent crash
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def train(self, data_loader, max_steps: int, checkpoint_dir: str, 
              save_every: int = 1000, eval_every: int = 500, log_every: int = 100):
        """
        Main training loop
        
        Args:
            data_loader: DataLoader or streaming dataset
            max_steps: Maximum training steps
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N steps
            eval_every: Evaluate every N steps
            log_every: Log metrics every N steps
        """
        logger.info(f"Starting training for {max_steps} steps...")
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
        
        self.model.train()
        
        # Setup progress bar
        progress_bar = tqdm(total=max_steps, desc="Training", initial=self.step)
        
        # Training metrics
        running_loss = 0.0
        step_times = []
        
        # Handle streaming vs regular dataloader
        if hasattr(data_loader, '__iter__') and not hasattr(data_loader, '__len__'):
            # Streaming dataset
            data_iter = iter(data_loader)
        else:
            # Regular DataLoader
            data_iter = iter(data_loader)
        
        try:
            while self.step < max_steps:
                step_start_time = time.time()
                
                # Get next batch
                try:
                    if hasattr(data_loader, '__iter__') and not hasattr(data_loader, '__len__'):
                        # Streaming
                        batch = next(data_iter)
                    else:
                        # Regular DataLoader
                        try:
                            batch = next(data_iter)
                        except StopIteration:
                            # Reset data loader
                            data_iter = iter(data_loader)
                            batch = next(data_iter)
                            self.epoch += 1
                except StopIteration:
                    logger.warning("Data iterator exhausted unexpectedly")
                    break
                
                # Training step
                loss = self.train_step(batch)
                running_loss += loss
                
                # Gradient accumulation
                if (self.step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Timing
                step_time = time.time() - step_start_time
                step_times.append(step_time)
                if len(step_times) > 100:
                    step_times.pop(0)
                
                # Logging
                if (self.step + 1) % log_every == 0:
                    avg_loss = running_loss / log_every
                    avg_time = sum(step_times) / len(step_times)
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    # GPU memory usage
                    gpu_memory = ""
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1e9
                        cached = torch.cuda.memory_reserved() / 1e9
                        gpu_memory = f"GPU: {allocated:.1f}GB/{cached:.1f}GB"
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'time/step': f'{avg_time:.2f}s',
                        'mem': gpu_memory
                    })
                    
                    # Log to console
                    logger.info(f"Step {self.step + 1}: Loss={avg_loss:.4f}, LR={current_lr:.2e}, {gpu_memory}")
                    
                    running_loss = 0.0
                
                # Checkpointing
                if (self.step + 1) % save_every == 0:
                    self._save_checkpoint(checkpoint_dir, self.step + 1, avg_loss if 'avg_loss' in locals() else None)
                
                self.step += 1
                progress_bar.update(1)
                
                # Early stopping for Kaggle time limits
                if self.step >= max_steps:
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always save final checkpoint
            self._save_checkpoint(checkpoint_dir, self.step, 
                                avg_loss if 'avg_loss' in locals() else None)
            progress_bar.close()
        
        logger.info("Training completed!")
    
    def _save_checkpoint(self, checkpoint_dir: str, step: int, loss: Optional[float] = None):
        """Save checkpoint (this will be implemented in checkpoint_manager.py)"""
        # This is a placeholder - the actual implementation will be in CheckpointManager
        logger.info(f"Checkpoint save requested for step {step}")
    
    def evaluate(self, eval_data_loader, max_eval_steps: int = 100) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            eval_data_loader: Evaluation data loader
            max_eval_steps: Maximum evaluation steps
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for step, batch in enumerate(eval_data_loader):
                if step >= max_eval_steps:
                    break
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        loss = self._forward_pass(input_ids, attention_mask, labels)
                else:
                    loss = self._forward_pass(input_ids, attention_mask, labels)
                
                total_loss += loss.item()
                num_samples += input_ids.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / max(num_samples, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Clip to prevent overflow
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_samples': num_samples
        }
        
        logger.info(f"Evaluation results: {metrics}")
        
        self.model.train()  # Switch back to training mode
        return metrics
    
    def generate_sample(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        Generate a sample from the model
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        try:
            # Tokenize prompt
            tokens = self.tokenizer.encode(prompt, add_bos=True)
            input_ids = torch.tensor([tokens], device=self.device)
            
            # Generate (this is a simplified version)
            with torch.no_grad():
                for _ in range(max_length):
                    # Get model output
                    outputs = self.model(input_ids)
                    
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, torch.Tensor):
                        logits = outputs
                    else:
                        logits = outputs[0]
                    
                    # Get next token probabilities
                    next_token_logits = logits[0, -1, :] / temperature
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)
                    
                    # Sample next token
                    next_token = torch.multinomial(next_token_probs, num_samples=1)
                    
                    # Append to input
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                    
                    # Stop if EOS token
                    if next_token.item() == self.tokenizer.eos_id:
                        break
            
            # Decode generated sequence
            generated_tokens = input_ids[0].tolist()
            generated_text = self.tokenizer.decode(generated_tokens)
            
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            generated_text = f"Generation failed: {e}"
        finally:
            self.model.train()
        
        return generated_text

if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("To use, instantiate GemmaTrainer with your model, tokenizer, and config")

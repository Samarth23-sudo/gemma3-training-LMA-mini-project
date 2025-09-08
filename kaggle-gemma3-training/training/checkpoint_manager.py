import torch
import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manage model checkpoints for Kaggle environment with automatic cleanup
    """
    
    def __init__(self, model, optimizer, scheduler, max_checkpoints: int = 3):
        """
        Initialize checkpoint manager
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            scheduler: Learning rate scheduler to checkpoint
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_checkpoints = max_checkpoints
        
        # Track saved checkpoints for cleanup
        self.saved_checkpoints = []
        
        logger.info(f"CheckpointManager initialized with max_checkpoints={max_checkpoints}")
    
    def save_checkpoint(self, checkpoint_dir: str, step: int, loss: Optional[float] = None,
                       metrics: Optional[Dict[str, Any]] = None, is_best: bool = False) -> str:
        """
        Save model checkpoint
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            step: Current training step
            loss: Current loss value
            metrics: Additional metrics to save
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'metrics': metrics or {},
            'model_config': getattr(self.model, 'config', None)
        }
        
        # Checkpoint filename
        checkpoint_filename = f'checkpoint_step_{step}.pt'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        
        try:
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Track saved checkpoint
            self.saved_checkpoints.append({
                'path': checkpoint_path,
                'step': step,
                'loss': loss,
                'is_best': is_best
            })
            
            # Save metadata
            self._save_metadata(checkpoint_dir, step, loss, metrics, checkpoint_path)
            
            # Save best checkpoint separately if needed
            if is_best:
                best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
                shutil.copy2(checkpoint_path, best_path)
                logger.info(f"Saved best checkpoint: {best_path}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(checkpoint_dir)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Log checkpoint info
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            logger.info(f"Checkpoint size: {size_mb:.1f} MB")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return ""
    
    def _save_metadata(self, checkpoint_dir: str, step: int, loss: Optional[float],
                      metrics: Optional[Dict[str, Any]], checkpoint_path: str):
        """Save checkpoint metadata"""
        metadata = {
            'latest_step': step,
            'latest_loss': loss,
            'latest_checkpoint': checkpoint_path,
            'latest_metrics': metrics or {},
            'saved_checkpoints': [
                {
                    'step': cp['step'],
                    'loss': cp['loss'],
                    'path': cp['path'],
                    'is_best': cp['is_best']
                }
                for cp in self.saved_checkpoints
            ]
        }
        
        metadata_path = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: str):
        """Remove old checkpoints to save space"""
        if len(self.saved_checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by step (oldest first)
        self.saved_checkpoints.sort(key=lambda x: x['step'])
        
        # Remove oldest checkpoints (but keep best ones)
        checkpoints_to_remove = []
        while len(self.saved_checkpoints) > self.max_checkpoints:
            oldest = self.saved_checkpoints[0]
            
            # Don't remove best checkpoints
            if not oldest['is_best']:
                checkpoints_to_remove.append(oldest)
                self.saved_checkpoints.remove(oldest)
            else:
                # If the oldest is best, remove the second oldest non-best
                for i, cp in enumerate(self.saved_checkpoints[1:], 1):
                    if not cp['is_best']:
                        checkpoints_to_remove.append(cp)
                        self.saved_checkpoints.remove(cp)
                        break
                else:
                    # All are best, remove oldest anyway
                    checkpoints_to_remove.append(oldest)
                    self.saved_checkpoints.remove(oldest)
        
        # Actually delete the files
        for cp in checkpoints_to_remove:
            try:
                if os.path.exists(cp['path']):
                    os.remove(cp['path'])
                    logger.info(f"Removed old checkpoint: {cp['path']}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {cp['path']}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> tuple[int, Optional[float]]:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (step, loss)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            step = checkpoint.get('step', 0)
            loss = checkpoint.get('loss', None)
            
            logger.info(f"Checkpoint loaded successfully - Step: {step}, Loss: {loss}")
            
            return step, loss
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """
        Find the latest checkpoint in a directory
        
        Args:
            checkpoint_dir: Directory to search
            
        Returns:
            Path to latest checkpoint or None
        """
        if not os.path.exists(checkpoint_dir):
            return None
        
        # Try to load from metadata first
        metadata_path = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    latest_checkpoint = metadata.get('latest_checkpoint')
                    if latest_checkpoint and os.path.exists(latest_checkpoint):
                        return latest_checkpoint
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        # Fallback: search for checkpoint files
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith('checkpoint_step_') and file.endswith('.pt'):
                try:
                    step = int(file.replace('checkpoint_step_', '').replace('.pt', ''))
                    checkpoint_files.append((step, os.path.join(checkpoint_dir, file)))
                except ValueError:
                    continue
        
        if checkpoint_files:
            # Return the checkpoint with highest step number
            checkpoint_files.sort(key=lambda x: x[0], reverse=True)
            return checkpoint_files[0][1]
        
        return None
    
    def find_best_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """
        Find the best checkpoint (lowest loss)
        
        Args:
            checkpoint_dir: Directory to search
            
        Returns:
            Path to best checkpoint or None
        """
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
        if os.path.exists(best_path):
            return best_path
        
        # Fallback: look through metadata
        metadata_path = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                best_checkpoint = None
                best_loss = float('inf')
                
                for cp in metadata.get('saved_checkpoints', []):
                    if cp.get('loss') is not None and cp['loss'] < best_loss:
                        best_loss = cp['loss']
                        best_checkpoint = cp['path']
                
                return best_checkpoint
                
            except Exception as e:
                logger.warning(f"Failed to find best checkpoint from metadata: {e}")
        
        return None
    
    def export_for_kaggle_dataset(self, checkpoint_dir: str, output_path: str, 
                                 include_optimizer: bool = False):
        """
        Export checkpoint for Kaggle dataset upload
        
        Args:
            checkpoint_dir: Source checkpoint directory
            output_path: Output path for exported checkpoint
            include_optimizer: Whether to include optimizer state
        """
        # Find best checkpoint
        best_checkpoint_path = self.find_best_checkpoint(checkpoint_dir)
        if not best_checkpoint_path:
            logger.warning("No best checkpoint found, using latest")
            best_checkpoint_path = self.find_latest_checkpoint(checkpoint_dir)
        
        if not best_checkpoint_path:
            raise ValueError("No checkpoint found to export")
        
        logger.info(f"Exporting checkpoint: {best_checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
        
        # Create export checkpoint with minimal data
        export_checkpoint = {
            'model_state_dict': checkpoint['model_state_dict'],
            'step': checkpoint.get('step', 0),
            'loss': checkpoint.get('loss'),
            'metrics': checkpoint.get('metrics', {}),
            'model_config': checkpoint.get('model_config')
        }
        
        # Optionally include optimizer state
        if include_optimizer:
            export_checkpoint['optimizer_state_dict'] = checkpoint.get('optimizer_state_dict')
            export_checkpoint['scheduler_state_dict'] = checkpoint.get('scheduler_state_dict')
        
        # Save export checkpoint
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(export_checkpoint, output_path)
        
        # Create info file
        info = {
            'export_info': {
                'source_checkpoint': best_checkpoint_path,
                'step': checkpoint.get('step', 0),
                'loss': checkpoint.get('loss'),
                'metrics': checkpoint.get('metrics', {}),
                'includes_optimizer': include_optimizer,
                'export_timestamp': str(torch.tensor(0).item())  # Placeholder
            }
        }
        
        info_path = output_path.replace('.pt', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Checkpoint exported to: {output_path} ({size_mb:.1f} MB)")
        
        return output_path
    
    def get_checkpoint_info(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Get information about checkpoints in a directory
        
        Args:
            checkpoint_dir: Directory to analyze
            
        Returns:
            Dictionary with checkpoint information
        """
        if not os.path.exists(checkpoint_dir):
            return {'error': 'Directory does not exist'}
        
        info = {
            'checkpoint_dir': checkpoint_dir,
            'total_checkpoints': 0,
            'latest_checkpoint': None,
            'best_checkpoint': None,
            'total_size_mb': 0,
            'checkpoints': []
        }
        
        # Find all checkpoint files
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pt') and 'checkpoint' in file:
                file_path = os.path.join(checkpoint_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                
                info['checkpoints'].append({
                    'filename': file,
                    'path': file_path,
                    'size_mb': file_size
                })
                info['total_size_mb'] += file_size
                info['total_checkpoints'] += 1
        
        # Find latest and best
        info['latest_checkpoint'] = self.find_latest_checkpoint(checkpoint_dir)
        info['best_checkpoint'] = self.find_best_checkpoint(checkpoint_dir)
        
        return info

def create_emergency_checkpoint(model, step: int, output_path: str):
    """
    Create emergency checkpoint (minimal data for Kaggle interruptions)
    
    Args:
        model: Model to save
        step: Current step
        output_path: Output path
    """
    emergency_checkpoint = {
        'model_state_dict': model.state_dict(),
        'step': step,
        'emergency_save': True
    }
    
    torch.save(emergency_checkpoint, output_path)
    logger.info(f"Emergency checkpoint saved: {output_path}")

if __name__ == "__main__":
    print("CheckpointManager module loaded successfully!")
    print("To use, instantiate CheckpointManager with your model, optimizer, and scheduler")

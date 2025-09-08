import sentencepiece as spm
import torch
from typing import List, Union

class CustomGemmaTokenizer:
    """Adapter to use your trained SentencePiece tokenizer with Gemma models"""
    
    def __init__(self, tokenizer_path: str):
        """
        Initialize the custom tokenizer
        
        Args:
            tokenizer_path: Path to your trained SentencePiece model file
        """
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(tokenizer_path)
        
        # Map to Gemma tokenizer interface
        self.vocab_size = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id() if self.sp_model.bos_id() != -1 else 1
        self.eos_id = self.sp_model.eos_id() if self.sp_model.eos_id() != -1 else 2  
        self.pad_id = self.sp_model.pad_id() if self.sp_model.pad_id() != -1 else 0
        self.unk_id = self.sp_model.unk_id() if self.sp_model.unk_id() != -1 else 3
        
        print(f"Tokenizer loaded: vocab_size={self.vocab_size}")
        print(f"Special tokens - BOS: {self.bos_id}, EOS: {self.eos_id}, PAD: {self.pad_id}, UNK: {self.unk_id}")
        
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text to encode
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        if not isinstance(text, str):
            text = str(text)
            
        token_ids = self.sp_model.encode(text, out_type=int)
        
        if add_bos and self.bos_id != -1:
            token_ids = [self.bos_id] + token_ids
            
        if add_eos and self.eos_id != -1:
            token_ids = token_ids + [self.eos_id]
            
        return token_ids
        
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs or torch tensor
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Filter out special tokens
        filtered_ids = [tid for tid in token_ids if tid not in [self.pad_id, self.bos_id, self.eos_id]]
        
        return self.sp_model.decode(filtered_ids)
        
    def encode_batch(self, texts: List[str], max_length: int = None, padding: bool = True, 
                    add_bos: bool = True, add_eos: bool = False) -> torch.Tensor:
        """
        Encode a batch of texts
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length (if None, use longest in batch)
            padding: Whether to pad sequences to same length
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
            
        Returns:
            Tensor of shape (batch_size, max_length)
        """
        encoded_batch = []
        
        for text in texts:
            tokens = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            encoded_batch.append(tokens)
        
        if max_length is None:
            max_length = max(len(tokens) for tokens in encoded_batch)
            
        if padding:
            # Pad or truncate to max_length
            padded_batch = []
            for tokens in encoded_batch:
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                elif len(tokens) < max_length:
                    tokens = tokens + [self.pad_id] * (max_length - len(tokens))
                padded_batch.append(tokens)
            
            return torch.tensor(padded_batch, dtype=torch.long)
        else:
            return encoded_batch
    
    def create_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask (1 for real tokens, 0 for padding)
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Attention mask tensor
        """
        return (token_ids != self.pad_id).long()
    
    def get_vocab_item(self, token_id: int) -> str:
        """Get vocabulary item for a token ID"""
        return self.sp_model.id_to_piece(token_id)
    
    def get_token_id(self, token: str) -> int:
        """Get token ID for a vocabulary item"""
        return self.sp_model.piece_to_id(token)
    
    def test_tokenizer(self, test_texts: List[str] = None):
        """Test the tokenizer with sample texts"""
        if test_texts is None:
            test_texts = [
                "Hello, how are you?",  # English
                "नमस्ते, आप कैसे हैं?",   # Hindi  
                "तुमी कसो आसात?"         # Konkani (if available)
            ]
        
        print("🧪 Testing Custom Gemma Tokenizer")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"Test {i}: {text}")
            
            # Encode
            tokens = self.encode(text)
            print(f"Tokens: {tokens}")
            print(f"Token count: {len(tokens)}")
            
            # Decode
            decoded = self.decode(tokens)
            print(f"Decoded: {decoded}")
            
            # Check roundtrip
            roundtrip_ok = text.strip() == decoded.strip()
            print(f"Roundtrip OK: {roundtrip_ok}")
            
            print("-" * 30)
        
        # Test batch encoding
        print("Batch encoding test:")
        batch_tokens = self.encode_batch(test_texts, max_length=50, padding=True)
        print(f"Batch shape: {batch_tokens.shape}")
        print(f"Batch tokens:\n{batch_tokens}")
        
        # Test attention mask
        attention_mask = self.create_attention_mask(batch_tokens)
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Attention mask:\n{attention_mask}")
        
        print("✅ Tokenizer test completed!")

if __name__ == "__main__":
    # Test the tokenizer (you'll need to update the path)
    tokenizer_path = "/path/to/your/multilingual.model"  # Update this path
    
    try:
        tokenizer = CustomGemmaTokenizer(tokenizer_path)
        tokenizer.test_tokenizer()
    except Exception as e:
        print(f"Error testing tokenizer: {e}")
        print("Please update the tokenizer_path variable with the correct path to your model")

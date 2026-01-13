"""
Complete Transformer Implementation from Scratch

A full, educational implementation of the Transformer architecture
using only NumPy. Includes encoder, decoder, and training utilities.
"""

import numpy as np
from typing import Optional, Tuple, List


# =============================================================================
# Utility Functions
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization.
    
    LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


# =============================================================================
# Positional Encoding
# =============================================================================

class SinusoidalPositionalEncoding:
    """Sinusoidal positional encoding from the original Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.pe = self._create_pe(max_len, d_model)
    
    def _create_pe(self, max_len: int, d_model: int) -> np.ndarray:
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    def __call__(self, seq_len: int) -> np.ndarray:
        return self.pe[:seq_len]


# =============================================================================
# Multi-Head Attention
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / np.sqrt(self.d_k)
        
        # Initialize weights
        self.W_Q = self._init_weight(d_model, d_model)
        self.W_K = self._init_weight(d_model, d_model)
        self.W_V = self._init_weight(d_model, d_model)
        self.W_O = self._init_weight(d_model, d_model)
        
        self.attention_weights = None
    
    def _init_weight(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier initialization."""
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * std
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_k)"""
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, n_heads, seq, d_k) -> (batch, seq, d_model)"""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq_len, self.d_model)
    
    def forward(
        self, 
        query: np.ndarray, 
        key: np.ndarray, 
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional (seq_q, seq_k), -inf for masked positions
            
        Returns:
            output: (batch, seq_q, d_model)
        """
        # Linear projections
        Q = query @ self.W_Q
        K = key @ self.W_K
        V = value @ self.W_V
        
        # Split heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Attention scores: (batch, n_heads, seq_q, seq_k)
        scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
        
        # Attention weights
        attn_weights = softmax(scores, axis=-1)
        self.attention_weights = attn_weights
        
        # Weighted sum
        attn_output = attn_weights @ V
        
        # Combine heads
        output = self._combine_heads(attn_output)
        
        # Output projection
        return output @ self.W_O


# =============================================================================
# Feed-Forward Network
# =============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int = None):
        d_ff = d_ff or 4 * d_model
        
        self.W1 = self._init_weight(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = self._init_weight(d_ff, d_model)
        self.b2 = np.zeros(d_model)
    
    def _init_weight(self, fan_in: int, fan_out: int) -> np.ndarray:
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * std
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = gelu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


# =============================================================================
# Transformer Encoder Block
# =============================================================================

class EncoderBlock:
    """
    Single Transformer Encoder Block.
    
    x -> LayerNorm -> Self-Attention -> + -> LayerNorm -> FFN -> +
         |__________________________|     |__________________|
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        
        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Pre-LayerNorm self-attention
        x_norm = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attn_out = self.attention.forward(x_norm, x_norm, x_norm, mask)
        x = x + attn_out
        
        # Pre-LayerNorm FFN
        x_norm = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ffn_out = self.ffn.forward(x_norm)
        x = x + ffn_out
        
        return x


# =============================================================================
# Transformer Decoder Block
# =============================================================================

class DecoderBlock:
    """
    Single Transformer Decoder Block.
    
    Includes:
    1. Masked self-attention
    2. Cross-attention (to encoder output)
    3. Feed-forward network
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        
        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)
        self.ln3_gamma = np.ones(d_model)
        self.ln3_beta = np.zeros(d_model)
    
    def forward(
        self, 
        x: np.ndarray, 
        encoder_output: np.ndarray,
        self_mask: Optional[np.ndarray] = None,
        cross_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # Masked self-attention
        x_norm = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        self_attn = self.self_attention.forward(x_norm, x_norm, x_norm, self_mask)
        x = x + self_attn
        
        # Cross-attention
        x_norm = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        cross_attn = self.cross_attention.forward(x_norm, encoder_output, encoder_output, cross_mask)
        x = x + cross_attn
        
        # FFN
        x_norm = layer_norm(x, self.ln3_gamma, self.ln3_beta)
        ffn_out = self.ffn.forward(x_norm)
        x = x + ffn_out
        
        return x


# =============================================================================
# Full Transformer
# =============================================================================

class Transformer:
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """
    
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_blocks = [
            EncoderBlock(d_model, n_heads, d_ff) 
            for _ in range(n_encoder_layers)
        ]
        self.encoder_ln_gamma = np.ones(d_model)
        self.encoder_ln_beta = np.zeros(d_model)
        
        # Decoder
        self.decoder_blocks = [
            DecoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_decoder_layers)
        ]
        self.decoder_ln_gamma = np.ones(d_model)
        self.decoder_ln_beta = np.zeros(d_model)
        
        # Output projection (tied with embedding)
        self.output_proj = self.embedding.T  # Weight tying
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal mask for decoder self-attention."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
        return mask
    
    def encode(self, src: np.ndarray, src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode source sequence.
        
        Args:
            src: Source token IDs (batch, src_len)
            src_mask: Padding mask
            
        Returns:
            encoder_output: (batch, src_len, d_model)
        """
        # Embed and add position
        x = self.embedding[src] * np.sqrt(self.d_model)
        x = x + self.pos_encoding(src.shape[1])
        
        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block.forward(x, src_mask)
        
        # Final layer norm
        x = layer_norm(x, self.encoder_ln_gamma, self.encoder_ln_beta)
        
        return x
    
    def decode(
        self, 
        tgt: np.ndarray, 
        encoder_output: np.ndarray,
        tgt_mask: Optional[np.ndarray] = None,
        cross_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Decode target sequence.
        
        Args:
            tgt: Target token IDs (batch, tgt_len)
            encoder_output: From encode() (batch, src_len, d_model)
            
        Returns:
            decoder_output: (batch, tgt_len, d_model)
        """
        seq_len = tgt.shape[1]
        
        # Embed and add position
        x = self.embedding[tgt] * np.sqrt(self.d_model)
        x = x + self.pos_encoding(seq_len)
        
        # Causal mask
        causal_mask = self._create_causal_mask(seq_len)
        if tgt_mask is not None:
            causal_mask = causal_mask + tgt_mask
        
        # Pass through decoder blocks
        for block in self.decoder_blocks:
            x = block.forward(x, encoder_output, causal_mask, cross_mask)
        
        # Final layer norm
        x = layer_norm(x, self.decoder_ln_gamma, self.decoder_ln_beta)
        
        return x
    
    def forward(
        self, 
        src: np.ndarray, 
        tgt: np.ndarray,
        src_mask: Optional[np.ndarray] = None,
        tgt_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Full forward pass.
        
        Args:
            src: Source tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)
            
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask)
        
        # Project to vocabulary
        logits = decoder_output @ self.output_proj
        
        return logits
    
    def generate(
        self, 
        src: np.ndarray, 
        max_len: int = 50,
        start_token: int = 1,
        end_token: int = 2
    ) -> np.ndarray:
        """
        Autoregressive generation.
        
        Args:
            src: Source tokens (batch, src_len)
            max_len: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            
        Returns:
            generated: Generated token IDs
        """
        batch_size = src.shape[0]
        encoder_output = self.encode(src)
        
        # Start with start token
        generated = np.full((batch_size, 1), start_token, dtype=np.int32)
        
        for _ in range(max_len):
            decoder_output = self.decode(generated, encoder_output)
            
            # Get logits for last position
            logits = decoder_output[:, -1, :] @ self.output_proj
            
            # Greedy: take argmax
            next_token = np.argmax(logits, axis=-1, keepdims=True)
            
            generated = np.concatenate([generated, next_token], axis=1)
            
            # Check for end token
            if np.all(next_token == end_token):
                break
        
        return generated


# =============================================================================
# Decoder-Only Transformer (GPT-style)
# =============================================================================

class GPTModel:
    """
    Decoder-only Transformer (GPT-style).
    
    Used for: text generation, language modeling
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 1024
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embedding = np.random.randn(max_len, d_model) * 0.01
        
        # Decoder blocks (with only self-attention)
        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append({
                'attention': MultiHeadAttention(d_model, n_heads),
                'ffn': FeedForward(d_model, d_ff),
                'ln1_gamma': np.ones(d_model),
                'ln1_beta': np.zeros(d_model),
                'ln2_gamma': np.ones(d_model),
                'ln2_beta': np.zeros(d_model),
            })
        
        # Final layer norm
        self.ln_f_gamma = np.ones(d_model)
        self.ln_f_beta = np.zeros(d_model)
        
        # Output (weight-tied with embedding)
        self.lm_head = self.token_embedding.T
    
    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            tokens: Token IDs (batch, seq_len)
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = tokens.shape
        
        # Embeddings
        x = self.token_embedding[tokens] + self.pos_embedding[:seq_len]
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
        
        # Transformer blocks
        for block in self.blocks:
            # Self-attention with residual
            x_norm = layer_norm(x, block['ln1_gamma'], block['ln1_beta'])
            attn = block['attention'].forward(x_norm, x_norm, x_norm, mask)
            x = x + attn
            
            # FFN with residual
            x_norm = layer_norm(x, block['ln2_gamma'], block['ln2_beta'])
            ffn = block['ffn'].forward(x_norm)
            x = x + ffn
        
        # Final layer norm
        x = layer_norm(x, self.ln_f_gamma, self.ln_f_beta)
        
        # Project to vocabulary
        logits = x @ self.lm_head
        
        return logits


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("Transformer Implementation from Scratch")
    print("=" * 60)
    
    # Parameters
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    n_layers = 2
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # Create model
    print("\n1. Creating Transformer model...")
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_layers,
        n_decoder_layers=n_layers
    )
    
    # Random input
    src = np.random.randint(0, vocab_size, (batch_size, src_len))
    tgt = np.random.randint(0, vocab_size, (batch_size, tgt_len))
    
    print(f"   Source shape: {src.shape}")
    print(f"   Target shape: {tgt.shape}")
    
    # Forward pass
    print("\n2. Forward pass...")
    logits = transformer.forward(src, tgt)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Expected: (batch={batch_size}, tgt_len={tgt_len}, vocab={vocab_size})")
    
    # GPT model
    print("\n3. Creating GPT-style model...")
    gpt = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )
    
    tokens = np.random.randint(0, vocab_size, (batch_size, 20))
    gpt_logits = gpt.forward(tokens)
    print(f"   Input: {tokens.shape}")
    print(f"   Output: {gpt_logits.shape}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

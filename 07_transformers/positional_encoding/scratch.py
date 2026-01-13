"""
Positional Encoding Implementations

Complete implementations of various positional encoding schemes:
1. Sinusoidal (original Transformer)
2. Learned embeddings
3. Rotary Position Embedding (RoPE)
4. ALiBi (Attention with Linear Biases)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class SinusoidalPositionalEncoding:
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need".
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        self.max_len = max_len
        
        # Precompute positional encodings
        self.pe = self._create_pe_matrix(max_len, d_model)
    
    def _create_pe_matrix(self, max_len: int, d_model: int) -> np.ndarray:
        """Create the positional encoding matrix."""
        pe = np.zeros((max_len, d_model))
        
        position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        
        # Compute the divisor term: 10000^(2i/d_model)
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )  # (d_model/2,)
        
        # Apply sin to even indices, cos to odd
        pe[:, 0::2] = np.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = np.cos(position * div_term)  # Odd dimensions
        
        return pe
    
    def encode(self, seq_len: int) -> np.ndarray:
        """
        Get positional encoding for given sequence length.
        
        Args:
            seq_len: Length of sequence
            
        Returns:
            Positional encoding matrix (seq_len, d_model)
        """
        return self.pe[:seq_len]
    
    def add_to_embeddings(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to token embeddings.
        
        Args:
            x: Token embeddings (batch, seq_len, d_model) or (seq_len, d_model)
            
        Returns:
            Embeddings with position information added
        """
        seq_len = x.shape[-2]
        return x + self.pe[:seq_len]
    
    def visualize(self, seq_len: int = 100, dims: int = None):
        """Visualize positional encodings."""
        dims = dims or self.d_model
        
        plt.figure(figsize=(12, 6))
        plt.imshow(self.pe[:seq_len, :dims], cmap='RdBu', aspect='auto')
        plt.colorbar(label='Encoding value')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Position')
        plt.title('Sinusoidal Positional Encoding')
        plt.tight_layout()
        plt.show()


class LearnedPositionalEmbedding:
    """
    Learned positional embeddings (used in BERT, GPT-2).
    
    Each position gets a learnable embedding vector.
    """
    
    def __init__(self, d_model: int, max_len: int = 512):
        """
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        self.max_len = max_len
        
        # Initialize embeddings (would be learned during training)
        self.embeddings = np.random.randn(max_len, d_model) * 0.02
    
    def encode(self, seq_len: int) -> np.ndarray:
        """Get position embeddings for given length."""
        if seq_len > self.max_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_len {self.max_len}")
        return self.embeddings[:seq_len]
    
    def add_to_embeddings(self, x: np.ndarray) -> np.ndarray:
        """Add position embeddings to token embeddings."""
        seq_len = x.shape[-2]
        return x + self.encode(seq_len)


class RotaryPositionalEmbedding:
    """
    Rotary Position Embedding (RoPE) from "RoFormer".
    
    Applies rotation to query/key vectors based on position.
    Used in LLaMA, GPT-NeoX, PaLM.
    """
    
    def __init__(self, d_model: int, base: float = 10000.0):
        """
        Args:
            d_model: Model dimension (must be even)
            base: Base for frequency computation
        """
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.base = base
        
        # Compute inverse frequencies
        # theta_i = 1 / (base^(2i/d))
        self.inv_freq = 1.0 / (
            base ** (np.arange(0, d_model, 2).astype(float) / d_model)
        )
    
    def _compute_rotary_embedding(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sin and cos matrices for RoPE.
        
        Returns:
            cos, sin: Each of shape (seq_len, d_model/2)
        """
        positions = np.arange(seq_len)
        
        # Outer product: (seq_len,) x (d_model/2,) -> (seq_len, d_model/2)
        freqs = np.outer(positions, self.inv_freq)
        
        return np.cos(freqs), np.sin(freqs)
    
    def apply(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """
        Apply rotary embeddings to input.
        
        Args:
            x: Input tensor (..., seq_len, d_model)
            start_pos: Starting position (for incremental decoding)
            
        Returns:
            Rotated tensor, same shape as input
        """
        seq_len = x.shape[-2]
        
        # Get cos and sin
        cos, sin = self._compute_rotary_embedding(start_pos + seq_len)
        cos = cos[start_pos:start_pos + seq_len]
        sin = sin[start_pos:start_pos + seq_len]
        
        # Split x into pairs: (x0, x1), (x2, x3), ...
        x1 = x[..., 0::2]  # Even dimensions
        x2 = x[..., 1::2]  # Odd dimensions
        
        # Apply rotation
        # [cos, -sin] [x1]   [x1*cos - x2*sin]
        # [sin,  cos] [x2] = [x1*sin + x2*cos]
        
        x_rotated = np.zeros_like(x)
        x_rotated[..., 0::2] = x1 * cos - x2 * sin
        x_rotated[..., 1::2] = x1 * sin + x2 * cos
        
        return x_rotated


class ALiBi:
    """
    Attention with Linear Biases (ALiBi).
    
    Adds linear penalty to attention scores based on distance.
    Used in BLOOM, MPT.
    """
    
    def __init__(self, n_heads: int):
        """
        Args:
            n_heads: Number of attention heads
        """
        self.n_heads = n_heads
        
        # Compute head-specific slopes
        # Slopes decrease geometrically: 2^(-8/n), 2^(-8*2/n), ...
        self.slopes = self._get_slopes(n_heads)
    
    def _get_slopes(self, n_heads: int) -> np.ndarray:
        """
        Compute ALiBi slopes for each head.
        
        For n_heads = 8: slopes = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(np.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if np.log2(n_heads) % 1 == 0:  # Power of 2
            return np.array(get_slopes_power_of_2(n_heads))
        else:
            # For non-power-of-2, interpolate
            closest_power = 2 ** int(np.log2(n_heads))
            return np.array(
                get_slopes_power_of_2(closest_power) + 
                get_slopes_power_of_2(2 * closest_power)[0::2][:n_heads - closest_power]
            )
    
    def get_bias(self, seq_len: int) -> np.ndarray:
        """
        Compute ALiBi bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Bias matrix (n_heads, seq_len, seq_len)
        """
        # Create distance matrix
        positions = np.arange(seq_len)
        distance = positions[np.newaxis, :] - positions[:, np.newaxis]  # (seq, seq)
        
        # Take absolute value (positions after current get positive penalty)
        distance = np.abs(distance)
        
        # Apply slopes: each head has different slope
        # Shape: (n_heads, seq_len, seq_len)
        bias = -distance[np.newaxis, :, :] * self.slopes[:, np.newaxis, np.newaxis]
        
        return bias
    
    def apply_to_attention(
        self, 
        attention_scores: np.ndarray, 
        causal: bool = True
    ) -> np.ndarray:
        """
        Apply ALiBi bias to attention scores.
        
        Args:
            attention_scores: (batch, n_heads, seq, seq)
            causal: If True, apply causal masking after bias
            
        Returns:
            Biased attention scores
        """
        seq_len = attention_scores.shape[-1]
        bias = self.get_bias(seq_len)
        
        # Add bias
        attention_scores = attention_scores + bias
        
        if causal:
            # Apply causal mask
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
            attention_scores = attention_scores + causal_mask
        
        return attention_scores


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Positional Encoding Implementations")
    print("=" * 60)
    
    d_model = 64
    seq_len = 20
    n_heads = 8
    
    # 1. Sinusoidal
    print("\n1. Sinusoidal Positional Encoding")
    print("-" * 40)
    
    sinusoidal = SinusoidalPositionalEncoding(d_model=d_model)
    pe = sinusoidal.encode(seq_len)
    print(f"   Shape: {pe.shape}")
    print(f"   Values bounded: [{pe.min():.3f}, {pe.max():.3f}]")
    
    # Show first few values
    print(f"\n   First 5 positions, first 8 dims:")
    print(pe[:5, :8].round(3))
    
    # 2. Learned
    print("\n2. Learned Positional Embedding")
    print("-" * 40)
    
    learned = LearnedPositionalEmbedding(d_model=d_model)
    le = learned.encode(seq_len)
    print(f"   Shape: {le.shape}")
    print(f"   (In practice, these would be trained)")
    
    # 3. RoPE
    print("\n3. Rotary Position Embedding (RoPE)")
    print("-" * 40)
    
    rope = RotaryPositionalEmbedding(d_model=d_model)
    x = np.random.randn(2, seq_len, d_model)  # Batch of 2
    x_rotated = rope.apply(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {x_rotated.shape}")
    print(f"   (Rotation preserves magnitude)")
    print(f"   Input norm: {np.linalg.norm(x[0, 0]):.4f}")
    print(f"   Output norm: {np.linalg.norm(x_rotated[0, 0]):.4f}")
    
    # 4. ALiBi
    print("\n4. ALiBi (Attention with Linear Biases)")
    print("-" * 40)
    
    alibi = ALiBi(n_heads=n_heads)
    bias = alibi.get_bias(seq_len=10)
    
    print(f"   Slopes: {alibi.slopes.round(4)}")
    print(f"   Bias shape: {bias.shape}")
    print(f"\n   Bias for head 0 (first 5x5):")
    print(bias[0, :5, :5].round(2))
    print(f"\n   (Negative values penalize distant tokens)")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

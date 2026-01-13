"""
Self-Attention Implementation from Scratch

Complete NumPy implementation with step-by-step mathematical correspondence.
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.
    
    softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
    
    For stability, we subtract max: softmax(x - max(x)) = softmax(x)
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Parameters:
    -----------
    d_k : int
        Dimension of queries and keys (for scaling)
    """
    
    def __init__(self, d_k: int):
        self.d_k = d_k
        self.scale = 1.0 / np.sqrt(d_k)
        
        # Store for backward pass
        self.attention_weights = None
    
    def forward(
        self, 
        Q: np.ndarray, 
        K: np.ndarray, 
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            Q: Queries, shape (batch, seq_len, d_k) or (seq_len, d_k)
            K: Keys, shape (batch, seq_len, d_k) or (seq_len, d_k)
            V: Values, shape (batch, seq_len, d_v) or (seq_len, d_v)
            mask: Optional mask, shape (seq_len, seq_len)
                  -inf for positions to mask, 0 otherwise
        
        Returns:
            output: Attention output, same shape as V
            attention_weights: Attention probabilities
        """
        # Step 1: Compute attention scores
        # S = QK^T, shape: (..., seq_len, seq_len)
        scores = np.matmul(Q, K.swapaxes(-2, -1))
        
        # Step 2: Scale by sqrt(d_k)
        # Prevents softmax saturation for large d_k
        scores = scores * self.scale
        
        # Step 3: Apply mask (for causal attention)
        if mask is not None:
            scores = scores + mask  # mask has -inf for masked positions
        
        # Step 4: Apply softmax to get attention weights
        # Each row sums to 1
        attention_weights = softmax(scores, axis=-1)
        self.attention_weights = attention_weights
        
        # Step 5: Weighted sum of values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights


class SelfAttention:
    """
    Single-Head Self-Attention Layer.
    
    Projects input to Q, K, V and applies scaled dot-product attention.
    """
    
    def __init__(self, d_model: int, d_k: int = None, d_v: int = None):
        """
        Args:
            d_model: Input/output embedding dimension
            d_k: Query/Key dimension (default: d_model)
            d_v: Value dimension (default: d_model)
        """
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        
        # Initialize projection matrices
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = np.random.randn(d_model, self.d_k) * scale
        self.W_K = np.random.randn(d_model, self.d_k) * scale
        self.W_V = np.random.randn(d_model, self.d_v) * scale
        self.W_O = np.random.randn(self.d_v, d_model) * scale
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(
        self, 
        x: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input, shape (batch, seq_len, d_model) or (seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: Same shape as input
        """
        # Project to Q, K, V
        Q = np.matmul(x, self.W_Q)  # (batch, seq, d_k)
        K = np.matmul(x, self.W_K)  # (batch, seq, d_k)
        V = np.matmul(x, self.W_V)  # (batch, seq, d_v)
        
        # Apply attention
        attn_output, _ = self.attention.forward(Q, K, V, mask)
        
        # Project back to d_model
        output = np.matmul(attn_output, self.W_O)
        
        return output


class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_v = d_model // n_heads
        
        # Combined projection matrices for efficiency
        # Shape: (d_model, d_model) but conceptually (d_model, n_heads * d_k)
        scale = np.sqrt(2.0 / (2 * d_model))
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
        self.attention = ScaledDotProductAttention(self.d_k)
        self.attention_weights = None
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split last dimension into (n_heads, d_k).
        
        (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        """
        batch_size = x.shape[0] if x.ndim == 3 else 1
        seq_len = x.shape[-2]
        
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)
    
    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back.
        
        (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        
        x = x.transpose(0, 2, 1, 3)  # (batch, seq, heads, d_k)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(
        self, 
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention: bool = False
    ) -> np.ndarray:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: If True, also return attention weights
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Handle 2D input
        squeeze_batch = False
        if x.ndim == 2:
            x = x[np.newaxis, ...]
            squeeze_batch = True
        
        # Linear projections
        Q = np.matmul(x, self.W_Q)
        K = np.matmul(x, self.W_K)
        V = np.matmul(x, self.W_V)
        
        # Split into heads
        Q = self._split_heads(Q)  # (batch, heads, seq, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Apply attention per head
        attn_output, attn_weights = self.attention.forward(Q, K, V, mask)
        self.attention_weights = attn_weights
        
        # Combine heads
        output = self._combine_heads(attn_output)
        
        # Final projection
        output = np.matmul(output, self.W_O)
        
        if squeeze_batch:
            output = output.squeeze(0)
        
        if return_attention:
            return output, attn_weights
        return output


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (look-ahead) mask for autoregressive models.
    
    Returns mask where position i cannot attend to positions > i.
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = mask * (-1e9)  # Large negative for softmax
    return mask


# =============================================================================
# Example Usage and Demonstration
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("Self-Attention Implementation Demo")
    print("=" * 60)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 64
    n_heads = 8
    
    # Create random input (simulating embeddings)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    
    # 1. Single-head attention
    print("\n" + "-" * 40)
    print("1. Single-Head Self-Attention")
    print("-" * 40)
    
    single_head = SelfAttention(d_model=d_model, d_k=64, d_v=64)
    output_single = single_head.forward(x[0])  # Single sample
    
    print(f"   Input:  {x[0].shape}")
    print(f"   Output: {output_single.shape}")
    
    # 2. Multi-head attention
    print("\n" + "-" * 40)
    print("2. Multi-Head Self-Attention")
    print("-" * 40)
    
    multi_head = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    output_multi, attn_weights = multi_head.forward(x, return_attention=True)
    
    print(f"   Input:  {x.shape}")
    print(f"   Output: {output_multi.shape}")
    print(f"   Attention weights: {attn_weights.shape}")
    print(f"   (batch={batch_size}, heads={n_heads}, seq={seq_len}, seq={seq_len})")
    
    # 3. Causal (masked) attention
    print("\n" + "-" * 40)
    print("3. Causal Self-Attention (for language modeling)")
    print("-" * 40)
    
    causal_mask = create_causal_mask(seq_len)
    output_causal = multi_head.forward(x, mask=causal_mask)
    
    print(f"   Mask shape: {causal_mask.shape}")
    print(f"   Output shape: {output_causal.shape}")
    print(f"   Causal attention prevents looking at future tokens")
    
    # 4. Visualize attention pattern
    print("\n" + "-" * 40)
    print("4. Attention Weight Analysis")
    print("-" * 40)
    
    # Get attention weights for first sample, first head
    attn = multi_head.attention_weights[0, 0]  # (seq, seq)
    
    print(f"   Attention for head 0, first 5x5 positions:")
    print(np.round(attn[:5, :5], 3))
    print(f"\n   Row sums (should be ~1.0): {attn.sum(axis=-1)[:5]}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

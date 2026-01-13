"""
Minimal GPT Implementation

A clean, educational GPT implementation for language modeling.
Inspired by Karpathy's nanoGPT.
"""

import numpy as np
from typing import Tuple, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class CausalSelfAttention:
    """Causal self-attention for autoregressive language modeling."""
    
    def __init__(self, d_model: int, n_heads: int, block_size: int):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        
        # Combined QKV projection
        self.c_attn = np.random.randn(d_model, 3 * d_model) * 0.02
        self.c_proj = np.random.randn(d_model, d_model) * 0.02
        
        # Causal mask
        self.mask = np.triu(np.ones((block_size, block_size)), k=1) * (-1e9)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, C = x.shape
        
        # QKV projection
        qkv = x @ self.c_attn  # (B, T, 3*C)
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # Reshape to heads: (B, T, C) -> (B, n_heads, T, d_head)
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # Attention
        scale = 1.0 / np.sqrt(self.d_head)
        att = (q @ k.transpose(0, 1, 3, 2)) * scale
        att = att + self.mask[:T, :T]
        att = softmax(att, axis=-1)
        
        # Weighted sum
        y = att @ v  # (B, n_heads, T, d_head)
        
        # Combine heads
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Output projection
        return y @ self.c_proj


class MLP:
    """Feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int):
        d_ff = 4 * d_model
        self.c_fc = np.random.randn(d_model, d_ff) * 0.02
        self.c_proj = np.random.randn(d_ff, d_model) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return gelu(x @ self.c_fc) @ self.c_proj


class Block:
    """Transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, block_size: int):
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.mlp = MLP(d_model)
        
        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn.forward(layer_norm(x, self.ln1_g, self.ln1_b))
        x = x + self.mlp.forward(layer_norm(x, self.ln2_g, self.ln2_b))
        return x


class GPT:
    """
    GPT Language Model.
    
    Architecture:
    - Token + Position Embeddings
    - N x Transformer Blocks
    - Layer Norm
    - Linear (weight-tied with embeddings)
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Embeddings
        self.wte = np.random.randn(vocab_size, n_embd) * 0.02  # Token
        self.wpe = np.random.randn(block_size, n_embd) * 0.01  # Position
        
        # Transformer blocks
        self.blocks = [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        
        # Final layer norm
        self.ln_f_g = np.ones(n_embd)
        self.ln_f_b = np.zeros(n_embd)
        
        # Output projection (weight-tied)
        self.lm_head = self.wte.T
    
    def forward(self, idx: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            idx: Token indices (B, T)
            
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"
        
        # Embed tokens and positions
        tok_emb = self.wte[idx]  # (B, T, n_embd)
        pos_emb = self.wpe[:T]   # (T, n_embd)
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Final norm and projection
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)
        logits = x @ self.lm_head
        
        return logits
    
    def generate(
        self, 
        idx: np.ndarray, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Autoregressive generation.
        
        Args:
            idx: Context tokens (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Extended sequence (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            
            # Get logits
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = np.sort(logits, axis=-1)[:, -top_k], np.argsort(logits, axis=-1)[:, -top_k:]
                logits[logits < v[:, [0]]] = -np.inf
            
            # Sample
            probs = softmax(logits, axis=-1)
            idx_next = np.array([np.random.choice(len(p), p=p) for p in probs])[:, np.newaxis]
            
            # Append
            idx = np.concatenate([idx, idx_next], axis=1)
        
        return idx


# =============================================================================
# Tokenizer (Character-level for simplicity)
# =============================================================================

class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text])
    
    def decode(self, tokens: np.ndarray) -> str:
        return ''.join([self.itos[int(i)] for i in tokens])


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("Minimal GPT Implementation")
    print("=" * 60)
    
    # Sample text
    text = """Machine learning is a subset of artificial intelligence.
    It enables computers to learn from data without explicit programming.
    Deep learning uses neural networks with many layers.
    Transformers are a key architecture for modern NLP."""
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {list(tokenizer.stoi.keys())[:20]}...")
    
    # Encode
    data = tokenizer.encode(text)
    print(f"\nEncoded shape: {data.shape}")
    
    # Create model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128
    )
    print(f"\nModel created:")
    print(f"  Block size: 64")
    print(f"  Layers: 4")
    print(f"  Heads: 4")
    print(f"  Embedding dim: 128")
    
    # Forward pass
    batch = data[:32][np.newaxis, :]
    logits = model.forward(batch)
    print(f"\nForward pass:")
    print(f"  Input: {batch.shape}")
    print(f"  Output: {logits.shape}")
    
    # Generate
    prompt = tokenizer.encode("Machine")
    print(f"\nGeneration (untrained, will be random):")
    print(f"  Prompt: 'Machine'")
    
    generated = model.generate(
        prompt[np.newaxis, :],
        max_new_tokens=20,
        temperature=1.0
    )
    print(f"  Generated: '{tokenizer.decode(generated[0])}'")
    
    print("\n" + "=" * 60)
    print("Note: This is an untrained model, so output is random.")
    print("Train the model on data to get meaningful generation!")
    print("=" * 60)

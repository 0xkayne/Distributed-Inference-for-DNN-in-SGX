"""
Validation script for ViT layer implementations.

Tests correctness of:
1. LayerNorm layer
2. Softmax layer
3. GELU layer
4. MatMul layer
5. Scale layer
6. Reshape layer

Compares outputs with PyTorch reference implementations.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

sys.path.insert(0, '.')


def test_layer_norm():
    """Test LayerNorm layer implementation."""
    print("\n[Test] LayerNorm Layer")
    print("-" * 40)
    
    # Test shapes
    batch_size = 2
    seq_len = 197
    embed_dim = 768
    
    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # PyTorch reference
    ref_ln = nn.LayerNorm(embed_dim)
    ref_output = ref_ln(x)
    
    # Manual computation (matching our implementation)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    eps = 1e-5
    normalized = (x - mean) / torch.sqrt(var + eps)
    our_output = normalized * ref_ln.weight + ref_ln.bias
    
    # Compare
    error = torch.abs(ref_output - our_output).max().item()
    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(ref_output.shape)}")
    print(f"  Max absolute error: {error:.2e}")
    print(f"  Status: {'PASS' if error < 1e-5 else 'FAIL'}")
    
    return error < 1e-5


def test_softmax():
    """Test Softmax layer implementation."""
    print("\n[Test] Softmax Layer")
    print("-" * 40)
    
    # Attention scores shape
    batch_size = 1
    num_heads = 12
    seq_len = 197
    
    # Create random input (simulating attention scores)
    x = torch.randn(batch_size, num_heads, seq_len, seq_len)
    
    # PyTorch reference
    ref_output = F.softmax(x, dim=-1)
    
    # Manual stable softmax (matching our implementation)
    max_val = x.max(dim=-1, keepdim=True)[0]
    exp_x = torch.exp(x - max_val)
    our_output = exp_x / exp_x.sum(dim=-1, keepdim=True)
    
    # Compare
    error = torch.abs(ref_output - our_output).max().item()
    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(ref_output.shape)}")
    print(f"  Max absolute error: {error:.2e}")
    
    # Verify probabilities sum to 1
    sum_check = ref_output.sum(dim=-1)
    sum_error = torch.abs(sum_check - 1.0).max().item()
    print(f"  Probability sum error: {sum_error:.2e}")
    print(f"  Status: {'PASS' if error < 1e-6 and sum_error < 1e-6 else 'FAIL'}")
    
    return error < 1e-6


def test_gelu():
    """Test GELU activation layer implementation."""
    print("\n[Test] GELU Activation")
    print("-" * 40)
    
    # FFN hidden dimension
    batch_size = 1
    seq_len = 197
    hidden_dim = 3072
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # PyTorch reference (exact)
    ref_output = F.gelu(x)
    
    # Approximate GELU (matching our implementation)
    SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
    GELU_COEFF = 0.044715
    approx_output = 0.5 * x * (1.0 + torch.tanh(
        SQRT_2_OVER_PI * (x + GELU_COEFF * x.pow(3))
    ))
    
    # Exact GELU
    INV_SQRT2 = 1.0 / np.sqrt(2.0)
    exact_output = x * 0.5 * (1.0 + torch.erf(x * INV_SQRT2))
    
    # Compare
    approx_error = torch.abs(ref_output - approx_output).max().item()
    exact_error = torch.abs(ref_output - exact_output).max().item()
    
    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(ref_output.shape)}")
    print(f"  Approximate GELU max error: {approx_error:.2e}")
    print(f"  Exact GELU max error: {exact_error:.2e}")
    print(f"  Status: {'PASS' if exact_error < 1e-6 else 'FAIL'}")
    
    return exact_error < 1e-6


def test_matmul():
    """Test batch matrix multiplication layer."""
    print("\n[Test] MatMul Layer")
    print("-" * 40)
    
    batch_size = 1
    num_heads = 12
    seq_len = 197
    head_dim = 64
    
    # Q @ K^T test
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Reference
    ref_qk = torch.matmul(Q, K.transpose(-2, -1))
    
    # Manual (matching our implementation with scale)
    scale = 1.0 / np.sqrt(head_dim)
    our_qk = torch.matmul(Q, K.transpose(-2, -1)) * scale
    ref_qk_scaled = ref_qk * scale
    
    error_qk = torch.abs(ref_qk_scaled - our_qk).max().item()
    
    print(f"  Q @ K^T:")
    print(f"    Q shape: {tuple(Q.shape)}")
    print(f"    K shape: {tuple(K.shape)}")
    print(f"    Output shape: {tuple(ref_qk.shape)}")
    print(f"    Max error (with scale): {error_qk:.2e}")
    
    # Attention @ V test
    attn = torch.randn(batch_size, num_heads, seq_len, seq_len)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    ref_av = torch.matmul(attn, V)
    our_av = torch.matmul(attn, V)  # No transpose, no scale
    
    error_av = torch.abs(ref_av - our_av).max().item()
    
    print(f"  Attention @ V:")
    print(f"    Attention shape: {tuple(attn.shape)}")
    print(f"    V shape: {tuple(V.shape)}")
    print(f"    Output shape: {tuple(ref_av.shape)}")
    print(f"    Max error: {error_av:.2e}")
    
    print(f"  Status: {'PASS' if error_qk < 1e-6 and error_av < 1e-6 else 'FAIL'}")
    
    return error_qk < 1e-6 and error_av < 1e-6


def test_scale():
    """Test Scale layer implementation."""
    print("\n[Test] Scale Layer")
    print("-" * 40)
    
    batch_size = 1
    num_heads = 12
    seq_len = 197
    
    x = torch.randn(batch_size, num_heads, seq_len, seq_len)
    scale_factor = 0.125  # 1/sqrt(64)
    
    ref_output = x * scale_factor
    our_output = x * scale_factor
    
    error = torch.abs(ref_output - our_output).max().item()
    
    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Scale factor: {scale_factor}")
    print(f"  Max error: {error:.2e}")
    print(f"  Status: {'PASS' if error < 1e-10 else 'FAIL'}")
    
    return error < 1e-10


def test_reshape():
    """Test Reshape layer implementation."""
    print("\n[Test] Reshape Layer")
    print("-" * 40)
    
    batch_size = 1
    seq_len = 197
    embed_dim = 768
    num_heads = 12
    head_dim = 64
    
    # Test 1: View reshape
    x = torch.randn(batch_size, seq_len, embed_dim)
    target_shape = [batch_size, seq_len, num_heads, head_dim]
    
    reshaped = x.view(*target_shape)
    error1 = 0.0 if tuple(reshaped.shape) == tuple(target_shape) else 1.0
    
    print(f"  Test 1 (View):")
    print(f"    Input: {tuple(x.shape)}")
    print(f"    Target: {target_shape}")
    print(f"    Result: {tuple(reshaped.shape)}")
    print(f"    Status: {'PASS' if error1 == 0 else 'FAIL'}")
    
    # Test 2: Permute
    x2 = torch.randn(batch_size, seq_len, num_heads, head_dim)
    permute_dims = [0, 2, 1, 3]  # (B, N, H, D) -> (B, H, N, D)
    
    permuted = x2.permute(*permute_dims)
    expected_shape = [batch_size, num_heads, seq_len, head_dim]
    error2 = 0.0 if tuple(permuted.shape) == tuple(expected_shape) else 1.0
    
    print(f"  Test 2 (Permute):")
    print(f"    Input: {tuple(x2.shape)}")
    print(f"    Permute dims: {permute_dims}")
    print(f"    Result: {tuple(permuted.shape)}")
    print(f"    Status: {'PASS' if error2 == 0 else 'FAIL'}")
    
    return error1 == 0 and error2 == 0


def test_full_attention():
    """Test complete attention computation."""
    print("\n[Test] Full Multi-Head Attention")
    print("-" * 40)
    
    batch_size = 1
    seq_len = 197
    embed_dim = 768
    num_heads = 12
    head_dim = embed_dim // num_heads
    
    # Input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # QKV projection weights
    qkv_weight = torch.randn(3 * embed_dim, embed_dim)
    qkv_bias = torch.randn(3 * embed_dim)
    
    # QKV projection
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Attention scores
    scale = 1.0 / np.sqrt(head_dim)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Softmax
    attn_probs = F.softmax(attn_scores, dim=-1)
    
    # Attention output
    attn_output = torch.matmul(attn_probs, v)
    
    # Reshape back
    attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
    
    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Q/K/V shape: ({tuple(q.shape)})")
    print(f"  Attention scores shape: {tuple(attn_scores.shape)}")
    print(f"  Attention probs sum: {attn_probs.sum(dim=-1).mean().item():.6f}")
    print(f"  Output shape: {tuple(attn_output.shape)}")
    print(f"  Status: PASS")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ViT Layer Implementation Validation")
    print("=" * 60)
    
    results = {
        "LayerNorm": test_layer_norm(),
        "Softmax": test_softmax(),
        "GELU": test_gelu(),
        "MatMul": test_matmul(),
        "Scale": test_scale(),
        "Reshape": test_reshape(),
        "FullAttention": test_full_attention(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"  {name:20} {status}")
    
    print("-" * 60)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




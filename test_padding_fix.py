#!/usr/bin/env python3
"""
Test script to demonstrate the padding mask bug fix.

This script creates a scenario where padding masks are used with streaming attention,
which previously would have caused shape mismatches.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pytorch.duoattention.block_streaming_attention import (
    construct_streaming_mask,
    block_streaming_attention
)


def test_construct_streaming_mask_with_padding():
    """Test construct_streaming_mask with single-sequence padding masks"""
    print("Testing construct_streaming_mask with padding masks...")
    
    device = torch.device('cpu')
    seqlen_q = 8
    seqlen_k = 10
    sink_size = 2
    local_size = 3
    
    # Create single-sequence padding masks (1D tensors)
    # First 6 tokens are valid, rest are padding
    query_mask = torch.cat([
        torch.ones(6, dtype=torch.bool, device=device),
        torch.zeros(2, dtype=torch.bool, device=device)
    ])
    
    # First 8 tokens are valid, rest are padding
    key_mask = torch.cat([
        torch.ones(8, dtype=torch.bool, device=device),
        torch.zeros(2, dtype=torch.bool, device=device)
    ])
    
    try:
        mask = construct_streaming_mask(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            sink_size=sink_size,
            local_size=local_size,
            is_causal=True,
            query_padding_mask=query_mask,
            key_padding_mask=key_mask,
            device=device
        )
        
        print(f"✓ Successfully created mask with shape: {mask.shape}")
        print(f"  Expected shape: ({seqlen_q}, {seqlen_k})")
        assert mask.shape == (seqlen_q, seqlen_k), "Shape mismatch!"
        print("✓ Shape is correct!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


def test_block_streaming_attention_with_padding():
    """Test block_streaming_attention with batch-level padding masks"""
    print("\nTesting block_streaming_attention with padding masks...")
    
    device = torch.device('cpu')
    batch_size = 2
    seqlens = [6, 8]  # Actual sequence lengths
    max_seqlen = 10   # Padded length
    num_heads = 2
    head_dim = 16
    sink_size = 2
    local_size = 3
    
    # Create padded input tensors
    total_len = batch_size * max_seqlen
    q = torch.randn(total_len, num_heads, head_dim, device=device)
    k = torch.randn(total_len, num_heads, head_dim, device=device)
    v = torch.randn(total_len, num_heads, head_dim, device=device)
    
    # Cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0, max_seqlen, 2 * max_seqlen], dtype=torch.int32, device=device)
    cu_seqlens_k = cu_seqlens_q.clone()
    
    # Create batch-level padding masks (batch_size, max_seqlen)
    query_padding_mask = torch.zeros(batch_size, max_seqlen, dtype=torch.bool, device=device)
    key_padding_mask = torch.zeros(batch_size, max_seqlen, dtype=torch.bool, device=device)
    
    # Mark valid positions as True
    for i, seqlen in enumerate(seqlens):
        query_padding_mask[i, :seqlen] = True
        key_padding_mask[i, :seqlen] = True
    
    # Use streaming attention for all heads
    head_mask_type = torch.full((num_heads,), -1, dtype=torch.int32, device=device)
    streaming_info = torch.tensor(
        [sink_size, local_size] * num_heads,
        dtype=torch.int32,
        device=device
    )
    
    try:
        output, _ = block_streaming_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            head_mask_type=head_mask_type,
            streaming_info=streaming_info,
            is_causal=True,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            return_attn_probs=False
        )
        
        print(f"✓ Successfully computed attention with shape: {output.shape}")
        print(f"  Expected shape: ({total_len}, {num_heads}, {head_dim})")
        assert output.shape == (total_len, num_heads, head_dim), "Shape mismatch!"
        print("✓ Shape is correct!")
        
        # Check for NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN!"
        assert not torch.isinf(output).any(), "Output contains Inf!"
        print("✓ Output is valid (no NaN or Inf)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Padding Mask Bug Fix")
    print("=" * 60)
    
    test_construct_streaming_mask_with_padding()
    test_block_streaming_attention_with_padding()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


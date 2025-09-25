#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimization utilities for StyleSync-AI.
Provides performance optimizations for the FLUX.1 Kontext model.
"""

import os
from typing import Dict, List, Optional, Union

import torch
from diffusers import FluxKontextPipeline


def optimize_pipeline_for_speed(
    pipe: FluxKontextPipeline,
    enable_xformers: bool = True,
    enable_sliced_attention: bool = False,
    enable_torch_compile: bool = False,
    enable_attention_slicing: bool = True,
    enable_cpu_offload: bool = False,
) -> FluxKontextPipeline:
    """
    Apply various optimizations to the pipeline for faster inference.
    
    Args:
        pipe: FLUX.1 Kontext pipeline to optimize
        enable_xformers: Whether to use xformers for memory efficient attention
        enable_sliced_attention: Whether to use sliced attention
        enable_torch_compile: Whether to use torch.compile (requires PyTorch 2.0+)
        enable_attention_slicing: Whether to use attention slicing
        enable_cpu_offload: Whether to offload modules to CPU when not in use
        
    Returns:
        FluxKontextPipeline: Optimized pipeline
    """
    # Use xFormers if available
    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✓ xFormers memory efficient attention enabled")
        except (ImportError, AttributeError):
            print("✗ xFormers not available, using default attention")
    
    # Use sliced attention as fallback if xFormers is not available
    if enable_sliced_attention and not enable_xformers:
        try:
            pipe.enable_attention_slicing(slice_size="auto")
            print("✓ Sliced attention enabled")
        except AttributeError:
            print("✗ Sliced attention not supported by this pipeline")
    
    # Use attention slicing
    if enable_attention_slicing:
        pipe.enable_attention_slicing()
        print("✓ Attention slicing enabled")
    
    # CPU offloading for lower VRAM usage
    if enable_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        print("✓ Sequential CPU offload enabled")
    
    # Try torch.compile for PyTorch 2.0+
    if enable_torch_compile and hasattr(torch, 'compile') and callable(torch.compile):
        try:
            # Compile the UNet model
            pipe.unet = torch.compile(
                pipe.unet, 
                mode="reduce-overhead", 
                fullgraph=True
            )
            print("✓ Model compiled with torch.compile")
        except Exception as e:
            print(f"✗ Failed to compile model: {e}")
    
    return pipe


def auto_optimize_for_device(
    pipe: FluxKontextPipeline,
    device: str = "auto",
    dtype: Optional[torch.dtype] = None,
) -> FluxKontextPipeline:
    """
    Automatically optimize the pipeline based on the available hardware.
    
    Args:
        pipe: FLUX.1 Kontext pipeline to optimize
        device: Device to optimize for ('auto', 'cuda', 'cpu', etc.)
        dtype: Data type to use for model weights
        
    Returns:
        FluxKontextPipeline: Optimized pipeline
    """
    # Detect device if set to auto
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine the optimal dtype if not specified
    if dtype is None:
        if device == "cuda":
            # Check for different precision support
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                # A100, H100, RTX 30xx, 40xx support bfloat16
                dtype = torch.bfloat16
                print("✓ Using bfloat16 precision (optimal for modern GPUs)")
            elif torch.cuda.is_available():
                # Older GPUs use float16
                dtype = torch.float16
                print("✓ Using float16 precision (optimal for older GPUs)")
            else:
                dtype = torch.float32
                print("✓ Using float32 precision (CUDA not available)")
        else:
            # CPU typically works best with float32
            dtype = torch.float32
            print("✓ Using float32 precision (optimal for CPU)")
    
    # Move pipeline to the device with the selected dtype
    pipe = pipe.to(device=device, dtype=dtype)
    
    # Apply optimizations based on device
    if device == "cuda":
        # GPU optimizations
        pipe = optimize_pipeline_for_speed(
            pipe,
            enable_xformers=True,
            enable_torch_compile=(torch.__version__ >= "2.0.0"),
            enable_attention_slicing=True,
        )
    else:
        # CPU optimizations (limited)
        print("✓ Running on CPU - limited optimization options available")
    
    return pipe


def use_tiled_vae(
    pipe: FluxKontextPipeline,
    tile_size: int = 512,
    tile_overlap: int = 32,
) -> FluxKontextPipeline:
    """
    Enable tiled VAE for processing large images with limited VRAM.
    
    Args:
        pipe: FLUX.1 Kontext pipeline
        tile_size: Size of the tiles to process
        tile_overlap: Overlap between tiles to prevent seams
        
    Returns:
        FluxKontextPipeline: Pipeline with tiled VAE enabled
    """
    try:
        pipe.enable_vae_tiling()
        print(f"✓ Tiled VAE enabled (tile_size={tile_size}, overlap={tile_overlap})")
        return pipe
    except AttributeError:
        print("✗ Tiled VAE not supported by this pipeline version")
        return pipe


def set_inference_batch_size(
    pipe: FluxKontextPipeline,
    batch_size: int = 1,
) -> FluxKontextPipeline:
    """
    Configure the pipeline for optimal batch processing.
    
    Args:
        pipe: FLUX.1 Kontext pipeline
        batch_size: Batch size for inference
        
    Returns:
        FluxKontextPipeline: Configured pipeline
    """
    # This is a placeholder for future batch configuration options
    # The actual batch size is handled in the pipeline call
    
    # Check if we can handle this batch size on the current hardware
    if pipe.device != "cpu" and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Rough estimate of VRAM needed per image (varies by model and settings)
        est_vram_per_image_gb = 7.0  # Estimated for FLUX.1 Kontext
        
        max_batch_size = int(vram_gb / est_vram_per_image_gb)
        
        if batch_size > max_batch_size:
            print(f"⚠️ Warning: Requested batch size {batch_size} may exceed available VRAM")
            print(f"   Estimated max batch size for current GPU: {max_batch_size}")
    
    return pipe
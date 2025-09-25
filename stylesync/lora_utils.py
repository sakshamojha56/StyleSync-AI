#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA utilities for StyleSync-AI.
Handles loading and managing LoRA weights for the FLUX.1 Kontext model.
"""

import os
from typing import Dict, List, Optional, Union

import torch
from diffusers import FluxKontextPipeline


def load_lora_weights(
    pipe: FluxKontextPipeline,
    lora_path: str,
    adapter_name: Optional[str] = None
) -> FluxKontextPipeline:
    """
    Load LoRA weights into the model pipeline.
    
    Args:
        pipe: FLUX.1 Kontext pipeline
        lora_path: Path to the LoRA weights (.safetensors file)
        adapter_name: Optional name for the adapter (for multi-LoRA)
        
    Returns:
        FluxKontextPipeline: Pipeline with loaded LoRA weights
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights file not found: {lora_path}")
    
    try:
        # For multi-LoRA support, use adapter_name
        if adapter_name:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        else:
            pipe.load_lora_weights(lora_path)
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load LoRA weights from {lora_path}: {e}")


def get_available_lora_styles(lora_dir: str) -> Dict[str, str]:
    """
    Get a dictionary of available LoRA style names and their file paths.
    
    Args:
        lora_dir: Directory containing LoRA weight files
        
    Returns:
        Dict[str, str]: Dictionary mapping style names to file paths
    """
    if not os.path.exists(lora_dir):
        return {}
    
    lora_files = {}
    
    for file in os.listdir(lora_dir):
        if file.endswith((".safetensors", ".pt", ".bin")):
            # Extract style name from filename
            style_name = os.path.splitext(file)[0].replace("_", " ").title()
            lora_files[style_name] = os.path.join(lora_dir, file)
    
    return lora_files


def load_multiple_loras(
    pipe: FluxKontextPipeline,
    lora_paths: List[str],
    weights: Optional[List[float]] = None,
    layer_weights: Optional[Dict[str, List[float]]] = None
) -> FluxKontextPipeline:
    """
    Load multiple LoRA weights with specified weights for blending.
    
    Args:
        pipe: FLUX.1 Kontext pipeline
        lora_paths: List of paths to LoRA weight files
        weights: Optional list of weights for each LoRA (must match length of lora_paths)
        layer_weights: Optional dictionary mapping layer types to weight lists for
                      fine-grained control over how LoRAs affect different parts of the model
        
    Returns:
        FluxKontextPipeline: Pipeline with loaded LoRA weights
    """
    if not lora_paths:
        return pipe
    
    # If weights not provided, use equal weighting
    if weights is None:
        weights = [1.0 / len(lora_paths)] * len(lora_paths)
    
    if len(weights) != len(lora_paths):
        raise ValueError("Number of weights must match number of LoRA paths")
    
    # Load each LoRA with a unique adapter name
    adapter_names = []
    for i, (path, weight) in enumerate(zip(lora_paths, weights)):
        adapter_name = f"lora_{i}"
        adapter_names.append(adapter_name)
        pipe = load_lora_weights(pipe, path, adapter_name=adapter_name)
    
    # Set the adapters with the specified weights
    if layer_weights:
        # Apply fine-grained layer-specific weights if provided
        pipe.set_adapters(
            adapter_names,
            adapter_weights=weights,
            layer_weights=layer_weights
        )
    else:
        # Otherwise use global weights
        pipe.set_adapters(adapter_names, adapter_weights=weights)
    
    return pipe


def create_layer_weight_config(
    text_encoder_weight: float = 1.0,
    unet_weight: float = 1.0,
    unet_blocks: Optional[Dict[str, float]] = None
) -> Dict[str, List[float]]:
    """
    Create a layer weight configuration for fine-grained LoRA application.
    
    Args:
        text_encoder_weight: Weight for text encoder layers
        unet_weight: Global weight for UNet layers
        unet_blocks: Optional dictionary mapping UNet block names to weights
                    for even more granular control
    
    Returns:
        Dict[str, List[float]]: Layer weight configuration
    """
    layer_weights = {
        "text_encoder": text_encoder_weight,
        "unet": unet_weight,
    }
    
    # Add per-block UNet weights if provided
    if unet_blocks:
        for block_name, weight in unet_blocks.items():
            layer_weights[f"unet.{block_name}"] = weight
    
    return layer_weights


def adjust_lora_scale(
    pipe: FluxKontextPipeline,
    scale: float
) -> FluxKontextPipeline:
    """
    Adjust the global scale of all loaded LoRAs.
    
    Args:
        pipe: FLUX.1 Kontext pipeline
        scale: Global scale factor for LoRA effects (0.0 to 1.0+)
        
    Returns:
        FluxKontextPipeline: Pipeline with adjusted LoRA scale
    """
    if hasattr(pipe, "set_lora_scale"):
        pipe.set_lora_scale(scale)
    else:
        # Fallback for older diffusers versions
        for module in pipe.modules():
            if hasattr(module, "scale"):
                module.scale = scale
    
    return pipe


def save_lora_blend(
    pipe: FluxKontextPipeline,
    output_path: str,
    adapter_names: List[str],
    weights: List[float]
) -> str:
    """
    Save a blend of multiple LoRAs as a new LoRA file.
    
    Args:
        pipe: FLUX.1 Kontext pipeline with loaded LoRAs
        output_path: Path to save the blended LoRA
        adapter_names: Names of the adapters to blend
        weights: Weights for each adapter in the blend
        
    Returns:
        str: Path to the saved blended LoRA
    """
    if not hasattr(pipe, "save_lora_weights"):
        raise NotImplementedError("Pipeline does not support saving LoRA weights")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the blend
    pipe.save_lora_weights(
        output_path,
        adapter_names=adapter_names,
        adapter_weights=weights,
    )
    
    return output_path
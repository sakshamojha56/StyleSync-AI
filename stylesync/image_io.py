#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image I/O utilities for StyleSync-AI.
Handles loading product images and saving generated marketing outputs.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from diffusers.utils import load_image


def load_product_image(image_path: str, height: int = 1024, width: int = 1024) -> Image.Image:
    """
    Load a product image from the specified path and resize if needed.
    
    Args:
        image_path: Path to the product image
        height: Target height for the image
        width: Target width for the image
        
    Returns:
        PIL.Image: Loaded and processed image
    """
    try:
        image = load_image(image_path)
        
        # Resize if needed while maintaining aspect ratio
        if image.height != height or image.width != width:
            image.thumbnail((width, height), Image.LANCZOS)
            
            # Create a new image with the target size and paste the resized image centered
            new_image = Image.new("RGB", (width, height), (255, 255, 255))
            paste_x = (width - image.width) // 2
            paste_y = (height - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
            
        return image
    except Exception as e:
        raise IOError(f"Failed to load image from {image_path}: {e}")


def save_generated_image(
    image: Image.Image, 
    output_dir: str, 
    product_name: str,
    scene_name: str,
    style_name: str,
    index: Optional[int] = None
) -> str:
    """
    Save a generated marketing image with descriptive filename.
    
    Args:
        image: Generated image to save
        output_dir: Directory to save the image to
        product_name: Name of the product
        scene_name: Name of the scene
        style_name: Name of the applied style
        index: Optional index for batch processing
        
    Returns:
        str: Path to the saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format scene and style names for filename
    scene_slug = scene_name.lower().replace(" ", "_")
    style_slug = style_name.lower().replace(" ", "_")
    product_slug = product_name.lower().replace(" ", "_")
    
    # Create filename
    if index is not None:
        filename = f"{product_slug}_{style_slug}_{scene_slug}_{index}.png"
    else:
        filename = f"{product_slug}_{style_slug}_{scene_slug}.png"
    
    file_path = os.path.join(output_dir, filename)
    
    # Save with high quality
    image.save(file_path, format="PNG")
    return file_path


def load_product_image_batch(image_paths: List[str], height: int = 1024, width: int = 1024) -> List[Image.Image]:
    """
    Load multiple product images for batch processing.
    
    Args:
        image_paths: List of paths to product images
        height: Target height for the images
        width: Target width for the images
        
    Returns:
        List[PIL.Image]: List of loaded and processed images
    """
    return [load_product_image(path, height, width) for path in image_paths]


def get_product_name_from_path(image_path: str) -> str:
    """
    Extract product name from image filename.
    
    Args:
        image_path: Path to the product image
        
    Returns:
        str: Extracted product name
    """
    filename = os.path.basename(image_path)
    product_name = os.path.splitext(filename)[0]
    return product_name.replace("_", " ").title()
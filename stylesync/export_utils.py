#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Output utilities for StyleSync-AI.
Provides customization options for saving and exporting generated images.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont, ExifTags

# Define common social media image sizes (width, height)
PLATFORM_SIZES = {
    # Social media platforms
    "instagram_post": (1080, 1080),
    "instagram_story": (1080, 1920),
    "facebook_post": (1200, 630),
    "facebook_cover": (1640, 924),
    "twitter_post": (1600, 900),
    "twitter_header": (1500, 500),
    "linkedin_post": (1200, 628),
    "linkedin_banner": (1584, 396),
    "pinterest_pin": (1000, 1500),
    "youtube_thumbnail": (1280, 720),
    "tiktok_video": (1080, 1920),
    
    # Digital advertising
    "display_small": (300, 250),    # Medium Rectangle
    "display_large": (728, 90),     # Leaderboard
    "display_wide": (970, 250),     # Billboard
    "mobile_banner": (320, 50),     # Mobile Banner
    "skyscraper": (160, 600),       # Wide Skyscraper
    
    # Print (at 300 DPI)
    "print_a4": (2480, 3508),       # A4 at 300 DPI
    "print_letter": (2550, 3300),   # Letter at 300 DPI
    "print_postcard": (1500, 1050), # 6"x4" at 300 DPI
    "print_business_card": (1050, 600), # 3.5"x2" at 300 DPI
    
    # Web
    "website_hero": (1600, 800),    # Website Hero Banner
    "website_header": (2000, 600),  # Website Header
    "website_banner": (1200, 300),  # Website Banner
}


def resize_for_platform(
    image: Image.Image,
    platform: str,
    crop: bool = False,
    fill_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Resize an image for a specific platform or use case.
    
    Args:
        image: Input image
        platform: Target platform/format name
        crop: Whether to crop the image to fit (True) or pad it (False)
        fill_color: Background color for padding
        
    Returns:
        PIL.Image: Resized image
    """
    if platform not in PLATFORM_SIZES:
        raise ValueError(f"Unknown platform: {platform}")
    
    target_width, target_height = PLATFORM_SIZES[platform]
    
    # Calculate aspect ratios
    img_aspect = image.width / image.height
    target_aspect = target_width / target_height
    
    if crop:
        # Crop approach: maintain target aspect ratio by cropping
        if img_aspect > target_aspect:
            # Image is wider than target, crop width
            new_width = int(image.height * target_aspect)
            left = (image.width - new_width) // 2
            image = image.crop((left, 0, left + new_width, image.height))
        elif img_aspect < target_aspect:
            # Image is taller than target, crop height
            new_height = int(image.width / target_aspect)
            top = (image.height - new_height) // 2
            image = image.crop((0, top, image.width, top + new_height))
            
        # Now resize to target dimensions
        image = image.resize((target_width, target_height), Image.LANCZOS)
        
    else:
        # Pad approach: resize to fit inside target and add padding
        if img_aspect > target_aspect:
            # Image is wider than target, fit to width
            new_width = target_width
            new_height = int(target_width / img_aspect)
        else:
            # Image is taller than target, fit to height
            new_height = target_height
            new_width = int(target_height * img_aspect)
        
        # Resize the image to new dimensions (fitting inside target)
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new image with target dimensions and paste the resized image centered
        new_img = Image.new('RGB', (target_width, target_height), fill_color)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_img.paste(resized_img, (paste_x, paste_y))
        
        image = new_img
    
    return image


def add_branding(
    image: Image.Image,
    logo_path: Optional[str] = None,
    watermark_text: Optional[str] = None,
    position: str = "bottom-right",
    opacity: float = 0.7,
    size_percent: float = 15.0,
) -> Image.Image:
    """
    Add branding elements like logo or watermark to an image.
    
    Args:
        image: Input image
        logo_path: Path to logo image file
        watermark_text: Text to use as watermark
        position: Position of branding ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
        opacity: Opacity of the branding element (0.0 to 1.0)
        size_percent: Size of logo as percentage of image width
        
    Returns:
        PIL.Image: Image with branding
    """
    result = image.copy()
    
    # Add logo if provided
    if logo_path and os.path.exists(logo_path):
        try:
            # Load and resize logo
            logo = Image.open(logo_path).convert("RGBA")
            
            # Calculate logo size based on percentage of image width
            logo_width = int(image.width * size_percent / 100)
            logo_height = int(logo.height * logo_width / logo.width)
            logo = logo.resize((logo_width, logo_height), Image.LANCZOS)
            
            # Apply opacity
            logo_data = logo.getdata()
            new_data = []
            for item in logo_data:
                # Apply opacity to alpha channel
                if item[3] > 0:  # If not completely transparent
                    new_data.append((item[0], item[1], item[2], int(item[3] * opacity)))
                else:
                    new_data.append(item)
            logo.putdata(new_data)
            
            # Determine position
            if position == "top-left":
                pos = (10, 10)
            elif position == "top-right":
                pos = (image.width - logo_width - 10, 10)
            elif position == "bottom-left":
                pos = (10, image.height - logo_height - 10)
            elif position == "bottom-right":
                pos = (image.width - logo_width - 10, image.height - logo_height - 10)
            elif position == "center":
                pos = ((image.width - logo_width) // 2, (image.height - logo_height) // 2)
            else:
                pos = (10, 10)  # Default to top-left
            
            # Paste logo onto image
            if result.mode != "RGBA":
                result = result.convert("RGBA")
            
            result.paste(logo, pos, logo)
            result = result.convert("RGB")  # Convert back to RGB if needed
        
        except Exception as e:
            print(f"Error adding logo: {e}")
    
    # Add watermark text if provided
    if watermark_text:
        try:
            # Create a drawing object
            draw = ImageDraw.Draw(result)
            
            # Try to use a nice font, fall back to default if not available
            try:
                font_size = int(image.width / 40)  # Scale font size with image
                font = ImageFont.truetype("Arial", font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size
            text_width, text_height = draw.textbbox((0, 0), watermark_text, font=font)[2:4]
            
            # Determine position
            if position == "top-left":
                pos = (10, 10)
            elif position == "top-right":
                pos = (image.width - text_width - 10, 10)
            elif position == "bottom-left":
                pos = (10, image.height - text_height - 10)
            elif position == "bottom-right":
                pos = (image.width - text_width - 10, image.height - text_height - 10)
            elif position == "center":
                pos = ((image.width - text_width) // 2, (image.height - text_height) // 2)
            else:
                pos = (10, 10)  # Default to top-left
            
            # Add semi-transparent background for better readability
            padding = 5
            draw.rectangle(
                [
                    pos[0] - padding, 
                    pos[1] - padding, 
                    pos[0] + text_width + padding, 
                    pos[1] + text_height + padding
                ],
                fill=(0, 0, 0, int(128 * opacity))
            )
            
            # Draw text
            draw.text(pos, watermark_text, fill=(255, 255, 255, int(255 * opacity)), font=font)
        
        except Exception as e:
            print(f"Error adding watermark text: {e}")
    
    return result


def add_metadata(
    image: Image.Image,
    metadata: Dict[str, str],
    embed_prompt: bool = True,
) -> Image.Image:
    """
    Add metadata to the image file.
    
    Args:
        image: Input image
        metadata: Dictionary of metadata to add
        embed_prompt: Whether to embed the prompt in the image metadata
        
    Returns:
        PIL.Image: Image with metadata
    """
    result = image.copy()
    
    # Add generation metadata as EXIF data
    exif_dict = {}
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exif_dict[0x9003] = timestamp  # DateTimeOriginal
    
    # Add software info
    exif_dict[0x0131] = "StyleSync-AI"  # Software
    
    # Add custom metadata
    for key, value in metadata.items():
        # Store in UserComment
        if key == "prompt" and embed_prompt:
            # 0x9286 is the tag for UserComment in EXIF
            exif_dict[0x9286] = f"prompt:{value}"
        
        # Other metadata will be added as XMP in future implementation
    
    # Add EXIF data to the image
    try:
        exif_bytes = result.getexif().tobytes()
        result.info["exif"] = exif_bytes
    except:
        # Fallback if EXIF not supported
        pass
    
    return result


def create_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    padding: int = 10,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create a grid of images for comparison.
    
    Args:
        images: List of images to include in the grid
        rows: Number of rows in the grid (calculated automatically if not specified)
        cols: Number of columns in the grid (calculated automatically if not specified)
        padding: Padding between images
        bg_color: Background color
        
    Returns:
        PIL.Image: Grid image
    """
    if not images:
        raise ValueError("No images provided to create grid")
    
    # Determine grid dimensions if not provided
    n_images = len(images)
    
    if rows is None and cols is None:
        # Auto-determine a reasonable grid
        cols = round(n_images ** 0.5)
        rows = (n_images + cols - 1) // cols  # Ceiling division
    elif rows is None:
        rows = (n_images + cols - 1) // cols  # Ceiling division
    elif cols is None:
        cols = (n_images + rows - 1) // rows  # Ceiling division
    
    # Ensure all images are the same size by resizing to the smallest dimensions
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    min_width = min(widths)
    min_height = min(heights)
    
    resized_images = []
    for img in images:
        if img.width != min_width or img.height != min_height:
            img = img.resize((min_width, min_height), Image.LANCZOS)
        resized_images.append(img)
    
    # Calculate the size of the output image
    grid_width = cols * min_width + (cols - 1) * padding
    grid_height = rows * min_height + (rows - 1) * padding
    
    # Create a new image for the grid
    grid_img = Image.new('RGB', (grid_width, grid_height), bg_color)
    
    # Paste images into the grid
    for idx, img in enumerate(resized_images[:rows * cols]):  # Limit to grid capacity
        row = idx // cols
        col = idx % cols
        
        x = col * (min_width + padding)
        y = row * (min_height + padding)
        
        grid_img.paste(img, (x, y))
    
    return grid_img


def export_for_platform(
    image: Image.Image,
    platform: str,
    output_dir: str,
    filename_prefix: str,
    metadata: Optional[Dict[str, str]] = None,
    add_branding_elements: bool = False,
    logo_path: Optional[str] = None,
    watermark_text: Optional[str] = None,
) -> str:
    """
    Export an image optimized for a specific platform.
    
    Args:
        image: Input image
        platform: Target platform/format name
        output_dir: Directory to save the exported image
        filename_prefix: Prefix for the output filename
        metadata: Optional metadata to embed in the image
        add_branding_elements: Whether to add branding elements
        logo_path: Path to logo image file
        watermark_text: Text to use as watermark
        
    Returns:
        str: Path to the exported image
    """
    # Create platform-specific directory
    platform_dir = os.path.join(output_dir, platform)
    os.makedirs(platform_dir, exist_ok=True)
    
    # Resize for the target platform
    resized_img = resize_for_platform(image, platform)
    
    # Add branding if requested
    if add_branding_elements:
        resized_img = add_branding(
            resized_img,
            logo_path=logo_path,
            watermark_text=watermark_text
        )
    
    # Add metadata if provided
    if metadata:
        resized_img = add_metadata(resized_img, metadata)
    
    # Create filename
    filename = f"{filename_prefix}_{platform}.png"
    filepath = os.path.join(platform_dir, filename)
    
    # Save the image
    resized_img.save(filepath, format="PNG")
    
    return filepath


def batch_export_for_platforms(
    image: Image.Image,
    platforms: List[str],
    output_dir: str,
    filename_prefix: str,
    metadata: Optional[Dict[str, str]] = None,
    add_branding_elements: bool = False,
    logo_path: Optional[str] = None,
    watermark_text: Optional[str] = None,
) -> List[str]:
    """
    Export an image optimized for multiple platforms.
    
    Args:
        image: Input image
        platforms: List of target platform/format names
        output_dir: Directory to save the exported images
        filename_prefix: Prefix for the output filenames
        metadata: Optional metadata to embed in the images
        add_branding_elements: Whether to add branding elements
        logo_path: Path to logo image file
        watermark_text: Text to use as watermark
        
    Returns:
        List[str]: Paths to the exported images
    """
    export_paths = []
    
    for platform in platforms:
        try:
            filepath = export_for_platform(
                image,
                platform,
                output_dir,
                filename_prefix,
                metadata,
                add_branding_elements,
                logo_path,
                watermark_text
            )
            export_paths.append(filepath)
        except Exception as e:
            print(f"Error exporting for platform {platform}: {e}")
    
    return export_paths


def get_available_platforms() -> Dict[str, Tuple[int, int]]:
    """
    Get a dictionary of available export platforms with their dimensions.
    
    Returns:
        Dict[str, Tuple[int, int]]: Dictionary mapping platform names to dimensions
    """
    return PLATFORM_SIZES
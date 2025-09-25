#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
StyleSync-AI: AI-Powered Contextual Product Marketing Workflow

Takes product images as input, applies branded visual styles using LoRA fine-tunes
on FLUX.1 Kontext dev, and places styled products into diverse marketing scenes.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers import FluxKontextPipeline
from PIL import Image
from tqdm import tqdm

# Import local modules
from stylesync import image_io, lora_utils, prompt_utils, optimization, export_utils


def load_flux_pipeline(
    device: str = "cuda", 
    torch_dtype: torch.dtype = torch.bfloat16,
    optimize: bool = True,
    use_xformers: bool = True,
    use_torch_compile: bool = False,
    cpu_offload: bool = False,
    tiled_vae: bool = False
) -> FluxKontextPipeline:
    """
    Load the FLUX.1 Kontext dev pipeline with optional optimizations.
    
    Args:
        device: Device to load the model on (cuda, cpu, etc.)
        torch_dtype: Torch data type for model precision
        optimize: Whether to automatically apply optimizations
        use_xformers: Whether to use xformers for memory efficient attention
        use_torch_compile: Whether to use torch.compile (requires PyTorch 2.0+)
        cpu_offload: Whether to offload modules to CPU when not in use
        tiled_vae: Whether to use tiled VAE for large images
        
    Returns:
        FluxKontextPipeline: Loaded and optimized pipeline
    """
    print(f"Loading FLUX.1 Kontext dev pipeline...")
    try:
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch_dtype,
        )
        
        if optimize:
            # Apply automatic optimizations based on device
            pipe = optimization.auto_optimize_for_device(pipe, device, torch_dtype)
        else:
            # Apply manual optimizations if specified
            pipe = pipe.to(device)
            
            if use_xformers:
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print("✓ xFormers memory efficient attention enabled")
                except:
                    print("✗ xFormers not available")
            
            if use_torch_compile and torch.__version__ >= "2.0.0":
                try:
                    pipe.unet = torch.compile(pipe.unet)
                    print("✓ Model compiled with torch.compile")
                except:
                    print("✗ torch.compile failed or not supported")
            
            if cpu_offload:
                pipe.enable_sequential_cpu_offload()
                print("✓ CPU offload enabled")
            
            if tiled_vae:
                optimization.use_tiled_vae(pipe)
        
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load FLUX.1 Kontext pipeline: {e}")


def process_single_image(
    pipe: FluxKontextPipeline,
    product_image: Image.Image,
    product_name: str,
    scene_type: str,
    scene_variant: str,
    style_name: str,
    output_dir: str,
    guidance_scale: float = 3.0,
    strength: float = 0.7,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    additional_context: Optional[str] = None,
    export_platforms: Optional[List[str]] = None,
    add_branding: bool = False,
    logo_path: Optional[str] = None,
    watermark_text: Optional[str] = None,
) -> Dict[str, Union[str, List[str]]]:
    """
    Process a single product image with the selected style and scene.
    
    Args:
        pipe: FLUX.1 Kontext pipeline
        product_image: Product image to process
        product_name: Name of the product
        scene_type: Type of marketing scene
        scene_variant: Variant of the scene
        style_name: Name of the style to apply
        output_dir: Directory to save output
        guidance_scale: Classifier-free guidance scale
        strength: Strength of the style transfer
        num_inference_steps: Number of inference steps
        seed: Random seed for reproducibility
        additional_context: Optional additional context for the prompt
        export_platforms: Optional list of platforms to export for
        add_branding: Whether to add branding elements
        logo_path: Path to logo image for branding
        watermark_text: Watermark text to add
        
    Returns:
        Dict[str, Union[str, List[str]]]: Paths to the saved images
    """
    # Build prompt
    prompt, negative_prompt = prompt_utils.build_marketing_prompt(
        product_description=product_name,
        scene_type=scene_type,
        scene_variant=scene_variant,
        style=style_name,
        additional_context=additional_context,
    )
    
    print(f"Generated prompt: {prompt}")
    
    # Set seed for reproducibility if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Generate image
    result = pipe(
        image=product_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    
    # Save main image
    saved_path = image_io.save_generated_image(
        image=result,
        output_dir=output_dir,
        product_name=product_name,
        scene_name=f"{scene_type}_{scene_variant}",
        style_name=style_name,
    )
    
    result_paths = {"main": saved_path, "exports": []}
    
    # Export for platforms if requested
    if export_platforms:
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "product": product_name,
            "scene": f"{scene_type}_{scene_variant}",
            "style": style_name,
            "guidance_scale": str(guidance_scale),
            "strength": str(strength),
            "steps": str(num_inference_steps),
            "seed": str(seed) if seed is not None else "None",
        }
        
        # Create filename prefix
        filename_prefix = f"{product_name}_{style_name}"
        
        # Export for each platform
        export_paths = export_utils.batch_export_for_platforms(
            image=result,
            platforms=export_platforms,
            output_dir=os.path.join(output_dir, "exports"),
            filename_prefix=filename_prefix,
            metadata=metadata,
            add_branding_elements=add_branding,
            logo_path=logo_path,
            watermark_text=watermark_text,
        )
        
        result_paths["exports"] = export_paths
    
    return result_paths


def batch_process(
    pipe: FluxKontextPipeline,
    product_paths: List[str],
    scene_types: List[str],
    scene_variants: Dict[str, List[str]],
    style_names: List[str],
    output_dir: str,
    guidance_scale: float = 3.0,
    strength: float = 0.7,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    additional_context: Optional[str] = None,
    export_platforms: Optional[List[str]] = None,
    add_branding: bool = False,
    logo_path: Optional[str] = None,
    watermark_text: Optional[str] = None,
    create_grid: bool = False,
) -> Dict[str, Union[List[str], str]]:
    """
    Batch process multiple product images with multiple styles and scenes.
    
    Args:
        pipe: FLUX.1 Kontext pipeline
        product_paths: List of paths to product images
        scene_types: List of scene types to use
        scene_variants: Dictionary mapping scene types to lists of variants
        style_names: List of style names to apply
        output_dir: Directory to save outputs
        guidance_scale: Classifier-free guidance scale
        strength: Strength of the style transfer
        num_inference_steps: Number of inference steps
        seed: Random seed for reproducibility
        additional_context: Optional additional context for the prompt
        export_platforms: Optional list of platforms to export for
        add_branding: Whether to add branding elements
        logo_path: Path to logo image for branding
        watermark_text: Watermark text to add
        create_grid: Whether to create a grid of all generated images
        
    Returns:
        Dict[str, Union[List[str], str]]: Paths to saved images and grids
    """
    result = {
        "main_images": [],
        "exported_images": [],
        "grids": []
    }
    
    # Calculate total number of images to generate
    total_images = len(product_paths) * sum(len(scene_variants.get(st, [])) for st in scene_types) * len(style_names)
    
    # Set up progress bar
    progress_bar = tqdm(total=total_images, desc="Generating marketing images")
    
    # Store generated images for grid creation
    all_generated_images = []
    
    # Process each product image
    for product_path in product_paths:
        product_image = image_io.load_product_image(product_path)
        product_name = image_io.get_product_name_from_path(product_path)
        
        # Store images for this product for potential grid creation
        product_images = []
        
        # Process each style
        for style_name in style_names:
            # Only load LoRA weights when style changes
            if hasattr(pipe, "lora_scale"):
                # Clear current LoRA weights if any
                pipe._lora_scale = None
            
            # Load new LoRA weights if available and not a built-in style
            if style_name in lora_utils.get_available_lora_styles("loras"):
                lora_path = lora_utils.get_available_lora_styles("loras")[style_name]
                pipe = lora_utils.load_lora_weights(pipe, lora_path)
            
            # Process each scene type and variant
            for scene_type in scene_types:
                variants = scene_variants.get(scene_type, [])
                if not variants:
                    # If no variants specified for this scene type, use all available
                    available_scenes = prompt_utils.get_available_scenes()
                    variants = available_scenes.get(scene_type, ["basic"])
                
                for scene_variant in variants:
                    result_paths = process_single_image(
                        pipe=pipe,
                        product_image=product_image,
                        product_name=product_name,
                        scene_type=scene_type,
                        scene_variant=scene_variant,
                        style_name=style_name,
                        output_dir=output_dir,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        seed=seed,
                        additional_context=additional_context,
                        export_platforms=export_platforms,
                        add_branding=add_branding,
                        logo_path=logo_path,
                        watermark_text=watermark_text,
                    )
                    
                    result["main_images"].append(result_paths["main"])
                    if "exports" in result_paths and result_paths["exports"]:
                        result["exported_images"].extend(result_paths["exports"])
                    
                    # Store image for grid
                    if create_grid:
                        img = Image.open(result_paths["main"])
                        product_images.append(img)
                        all_generated_images.append(img)
                    
                    progress_bar.update(1)
        
        # Create per-product grid if needed
        if create_grid and len(product_images) > 1:
            grid_path = os.path.join(output_dir, "grids", f"{product_name}_grid.png")
            os.makedirs(os.path.dirname(grid_path), exist_ok=True)
            
            grid = export_utils.create_image_grid(product_images)
            grid.save(grid_path)
            result["grids"].append(grid_path)
    
    progress_bar.close()
    
    # Create overall grid if needed
    if create_grid and len(all_generated_images) > 1:
        grid_path = os.path.join(output_dir, "grids", "all_images_grid.png")
        os.makedirs(os.path.dirname(grid_path), exist_ok=True)
        
        grid = export_utils.create_image_grid(all_generated_images)
        grid.save(grid_path)
        result["grids"].append(grid_path)
    
    return result


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="StyleSync-AI: Contextual Product Marketing Generator")
    
    # Input arguments
    parser.add_argument(
        "--product", "-p", 
        type=str, 
        nargs="+",
        help="Path(s) to product image(s)"
    )
    parser.add_argument(
        "--style", "-s", 
        type=str, 
        nargs="+", 
        default=["premium"],
        help="Style name(s) to apply to the product"
    )
    parser.add_argument(
        "--scene-type", "-st", 
        type=str, 
        nargs="+", 
        default=["billboard"],
        help="Type(s) of marketing scene to use"
    )
    parser.add_argument(
        "--scene-variant", "-sv", 
        type=str, 
        nargs="+", 
        default=["sunset"],
        help="Variant(s) of the scene type to use"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        type=str, 
        default="outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--lora-dir", 
        type=str, 
        default="loras",
        help="Directory containing LoRA weight files"
    )
    
    # Pipeline parameters
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to run the model on (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--guidance-scale", 
        type=float, 
        default=3.0,
        help="Guidance scale for generation"
    )
    parser.add_argument(
        "--strength", 
        type=float, 
        default=0.7,
        help="Strength of the style transfer"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=30,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--context", 
        type=str, 
        default=None,
        help="Additional context to add to the prompt"
    )
    
    # Advanced options
    parser.add_argument(
        "--multi-lora", 
        action="store_true",
        help="Enable multi-LoRA blending"
    )
    parser.add_argument(
        "--lora-weights", 
        type=float, 
        nargs="+", 
        default=None,
        help="Weights for multi-LoRA blending"
    )
    parser.add_argument(
        "--lora-scale", 
        type=float, 
        default=1.0,
        help="Global scale factor for LoRA effect strength"
    )
    parser.add_argument(
        "--custom-prompt", 
        type=str, 
        default=None,
        help="Custom prompt override"
    )
    parser.add_argument(
        "--list-scenes", 
        action="store_true",
        help="List available scene types and variants"
    )
    parser.add_argument(
        "--list-styles", 
        action="store_true",
        help="List available style templates"
    )
    
    # Optimization options
    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Model precision to use (auto, fp16, bf16, fp32)"
    )
    parser.add_argument(
        "--xformers",
        action="store_true",
        help="Enable xFormers memory efficient attention"
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile optimization (requires PyTorch 2.0+)"
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offloading for lower VRAM usage"
    )
    parser.add_argument(
        "--tiled-vae",
        action="store_true",
        help="Enable tiled VAE for large images with limited VRAM"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (use with caution)"
    )
    
    # Export options
    parser.add_argument(
        "--export-platforms",
        type=str,
        nargs="+",
        default=None,
        help="Export to specific platforms (e.g., instagram_post, facebook_post)"
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export to all available platforms"
    )
    parser.add_argument(
        "--add-branding",
        action="store_true",
        help="Add branding elements to exported images"
    )
    parser.add_argument(
        "--logo",
        type=str,
        default=None,
        help="Path to logo image for branding"
    )
    parser.add_argument(
        "--watermark",
        type=str,
        default=None,
        help="Watermark text to add to images"
    )
    parser.add_argument(
        "--create-grid",
        action="store_true",
        help="Create a grid of all generated images"
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="List all available export platforms and sizes"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Handle information listing commands
    if args.list_scenes:
        available_scenes = prompt_utils.get_available_scenes()
        print("\nAvailable Scene Types and Variants:")
        for scene_type, variants in available_scenes.items():
            print(f"  {scene_type}:")
            for variant in variants:
                print(f"    - {variant}")
        return
    
    if args.list_styles:
        available_styles = prompt_utils.get_available_styles()
        print("\nAvailable Style Templates:")
        for style in available_styles:
            print(f"  - {style}")
        return
        
    if args.list_platforms:
        available_platforms = export_utils.get_available_platforms()
        print("\nAvailable Export Platforms and Dimensions:")
        for platform, dimensions in available_platforms.items():
            print(f"  {platform}: {dimensions[0]}x{dimensions[1]} px")
        return
    
    # Ensure product paths are provided
    if not args.product:
        print("Error: At least one product image path must be provided.")
        return
    
    # Ensure product paths exist
    for path in args.product:
        if not os.path.exists(path):
            print(f"Error: Product image path does not exist: {path}")
            return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine torch dtype based on precision flag
    torch_dtype = None
    if args.precision == "fp16":
        torch_dtype = torch.float16
    elif args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp32":
        torch_dtype = torch.float32
    # else: auto will be handled by auto_optimize_for_device
    
    # Load pipeline with optimizations
    try:
        pipe = load_flux_pipeline(
            device=args.device, 
            torch_dtype=torch_dtype,
            optimize=(args.precision == "auto"),
            use_xformers=args.xformers,
            use_torch_compile=args.torch_compile,
            cpu_offload=args.cpu_offload,
            tiled_vae=args.tiled_vae
        )
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return
    
    # Prepare scene variants dictionary
    scene_variants = {}
    for i, scene_type in enumerate(args.scene_type):
        if i < len(args.scene_variant):
            scene_variants[scene_type] = [args.scene_variant[i]]
        else:
            # Default to "basic" if no variant specified
            scene_variants[scene_type] = ["basic"]
    
    # Handle custom prompt if provided
    if args.custom_prompt:
        product_image = image_io.load_product_image(args.product[0])
        product_name = image_io.get_product_name_from_path(args.product[0])
        
        prompt, negative_prompt = prompt_utils.build_custom_prompt(
            product_description=product_name,
            custom_scene=args.custom_prompt,
            style=args.style[0],
            additional_context=args.context,
        )
        
        # Set seed for reproducibility if provided
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        
        # Generate image
        result = pipe(
            image=product_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]
        
        # Save image
        saved_path = image_io.save_generated_image(
            image=result,
            output_dir=args.output_dir,
            product_name=product_name,
            scene_name="custom",
            style_name=args.style[0],
        )
        
        print(f"Generated image saved to: {saved_path}")
        return
    
    # Handle multi-LoRA if enabled
    if args.multi_lora and len(args.style) > 1:
        # Get LoRA paths
        lora_paths = []
        for style in args.style:
            if style in lora_utils.get_available_lora_styles(args.lora_dir):
                lora_paths.append(lora_utils.get_available_lora_styles(args.lora_dir)[style])
        
        if lora_paths:
            # Configure layer weights for fine-grained control (example)
            layer_weights = None
            if len(args.style) == 2:
                # Example configuration for a two-LoRA blend
                # First LoRA affects text encoder more, second affects UNet more
                layer_weights = {
                    "text_encoder": [0.8, 0.2],  # First LoRA has more influence on text
                    "unet": [0.3, 0.7],          # Second LoRA has more influence on image
                }
            
            # Load multiple LoRAs with weights
            pipe = lora_utils.load_multiple_loras(
                pipe, 
                lora_paths, 
                args.lora_weights,
                layer_weights
            )
            
    # Apply global LoRA scale if different from default
    if args.lora_scale != 1.0:
        pipe = lora_utils.adjust_lora_scale(pipe, args.lora_scale)
    
    # Prepare export platforms
    export_platforms = None
    if args.export_all:
        # Export to all platforms
        export_platforms = list(export_utils.get_available_platforms().keys())
    elif args.export_platforms:
        # Export to specified platforms
        export_platforms = args.export_platforms
        
        # Validate platforms
        available_platforms = export_utils.get_available_platforms()
        for platform in export_platforms:
            if platform not in available_platforms:
                print(f"Warning: Unknown platform '{platform}'. Run with --list-platforms to see available options.")
    
    # Start batch processing
    print(f"Processing {len(args.product)} product(s) with {len(args.style)} style(s) in {len(args.scene_type)} scene type(s)...")
    if export_platforms:
        print(f"Will export to {len(export_platforms)} platforms: {', '.join(export_platforms)}")
    
    start_time = time.time()
    
    results = batch_process(
        pipe=pipe,
        product_paths=args.product,
        scene_types=args.scene_type,
        scene_variants=scene_variants,
        style_names=args.style,
        output_dir=args.output_dir,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        num_inference_steps=args.steps,
        seed=args.seed,
        additional_context=args.context,
        export_platforms=export_platforms,
        add_branding=args.add_branding,
        logo_path=args.logo,
        watermark_text=args.watermark,
        create_grid=args.create_grid,
    )
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    main_count = len(results["main_images"])
    export_count = len(results["exported_images"])
    grid_count = len(results["grids"])
    
    print(f"\nGenerated {main_count} marketing images in {elapsed_time:.2f} seconds.")
    if export_count > 0:
        print(f"Created {export_count} platform-specific exports.")
    if grid_count > 0:
        print(f"Created {grid_count} image comparison grids.")
    
    print(f"Output directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
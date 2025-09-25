#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web UI for StyleSync-AI using Gradio.
Provides a user-friendly interface for non-technical users.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr
import torch
from PIL import Image

# Import local modules
from stylesync import image_io, lora_utils, prompt_utils, optimization, export_utils
from stylesync.main import load_flux_pipeline


class StyleSyncWebUI:
    def __init__(self, config=None):
        """Initialize the web UI with optional configuration."""
        self.pipe = None
        self.config = config or {}
        self.lora_dir = self.config.get("lora_dir", "loras")
        self.output_dir = self.config.get("output_dir", "outputs")
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get available scenes and styles
        self.available_scenes = prompt_utils.get_available_scenes()
        self.available_styles = prompt_utils.get_available_styles()
        
        # Get available LoRA models
        self.available_loras = lora_utils.get_available_lora_styles(self.lora_dir)
        
        # Combine built-in styles and LoRA models
        self.all_style_options = list(self.available_styles.keys()) + list(self.available_loras.keys())
        
        # Get available export platforms
        self.available_platforms = export_utils.get_available_platforms()
    
    def load_model(self, progress=gr.Progress()):
        """Load the model with progress updates."""
        if self.pipe is not None:
            return "Model already loaded"
        
        try:
            progress(0, desc="Loading model")
            
            # Determine torch dtype
            torch_dtype = torch.float16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                torch_dtype = torch.bfloat16
            
            progress(0.2, desc="Loading FLUX.1 Kontext dev pipeline")
            
            # Load pipeline with optimizations
            self.pipe = load_flux_pipeline(
                device=self.device,
                torch_dtype=torch_dtype,
                optimize=True,
                use_xformers=True,
            )
            
            progress(0.8, desc="Finalizing model")
            progress(1.0, desc="Model loaded")
            
            return "✅ Model loaded successfully"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def unload_model(self):
        """Unload the model to free GPU memory."""
        if self.pipe is None:
            return "Model not loaded"
        
        try:
            del self.pipe
            self.pipe = None
            
            # Force CUDA garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            return "✅ Model unloaded successfully"
        except Exception as e:
            return f"❌ Error unloading model: {str(e)}"
    
    def update_scene_variants(self, scene_type):
        """Update the scene variant dropdown options based on the selected scene type."""
        if not scene_type:
            return gr.Dropdown(choices=[])
        
        variants = self.available_scenes.get(scene_type, {})
        if not variants:
            return gr.Dropdown(choices=["basic"])
        
        return gr.Dropdown(choices=list(variants))
    
    def generate_image(
        self,
        product_image,
        product_description,
        style,
        scene_type,
        scene_variant,
        guidance_scale,
        strength,
        num_steps,
        additional_context,
        progress=gr.Progress()
    ):
        """Generate an image based on user inputs."""
        if self.pipe is None:
            return None, "❌ Model not loaded. Please load the model first."
        
        if product_image is None:
            return None, "❌ Please upload a product image."
        
        try:
            progress(0, desc="Starting generation")
            
            # Process the product image
            product_img = Image.fromarray(product_image)
            
            # Check if style is a LoRA model
            if style in self.available_loras:
                progress(0.1, desc="Loading LoRA weights")
                lora_path = self.available_loras[style]
                self.pipe = lora_utils.load_lora_weights(self.pipe, lora_path)
            
            # Build prompt
            progress(0.2, desc="Building prompt")
            prompt, negative_prompt = prompt_utils.build_marketing_prompt(
                product_description=product_description,
                scene_type=scene_type,
                scene_variant=scene_variant,
                style=style,
                additional_context=additional_context,
            )
            
            progress(0.3, desc=f"Generating image with {num_steps} steps")
            
            # Generate image
            result = self.pipe(
                image=product_img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                strength=strength,
                num_inference_steps=num_steps,
            ).images[0]
            
            progress(0.9, desc="Saving result")
            
            # Save the result
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            product_name = product_description.split()[0].lower() if product_description else "product"
            filename = f"{product_name}_{style}_{scene_type}_{scene_variant}_{timestamp}.png"
            output_path = os.path.join(self.output_dir, filename)
            result.save(output_path)
            
            progress(1.0, desc="Generation complete")
            
            # Return the result image and the prompt used
            log_message = f"✅ Generation successful\n\nPrompt: {prompt}\n\nNegative prompt: {negative_prompt}\n\nSaved to: {output_path}"
            return result, log_message
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return None, f"❌ Error during generation: {str(e)}\n\n{error_details}"
    
    def create_ui(self):
        """Create the Gradio UI for StyleSync-AI."""
        with gr.Blocks(title="StyleSync-AI") as ui:
            gr.Markdown("# StyleSync-AI: AI-Powered Product Marketing Image Generator")
            gr.Markdown("Create styled product images for various marketing contexts using FLUX.1 Kontext model with LoRA style transfer.")
            
            with gr.Row():
                with gr.Column():
                    # Model loading controls
                    with gr.Group():
                        gr.Markdown("### Model Controls")
                        with gr.Row():
                            load_btn = gr.Button("Load Model", variant="primary")
                            unload_btn = gr.Button("Unload Model", variant="secondary")
                        model_status = gr.Textbox(label="Model Status", interactive=False)
                    
                    # Input image and product description
                    with gr.Group():
                        gr.Markdown("### Product Input")
                        product_image = gr.Image(label="Product Image", type="numpy")
                        product_description = gr.Textbox(
                            label="Product Description", 
                            placeholder="e.g., premium leather sneakers, luxury watch",
                            value="premium product"
                        )
                    
                    # Style and scene selection
                    with gr.Group():
                        gr.Markdown("### Style and Scene")
                        style = gr.Dropdown(
                            label="Style",
                            choices=self.all_style_options,
                            value="luxury" if "luxury" in self.all_style_options else None
                        )
                        
                        scene_type = gr.Dropdown(
                            label="Scene Type",
                            choices=list(self.available_scenes.keys()),
                            value="billboard" if "billboard" in self.available_scenes else None
                        )
                        
                        scene_variant = gr.Dropdown(
                            label="Scene Variant",
                            choices=list(self.available_scenes.get("billboard", {}).keys()) if "billboard" in self.available_scenes else []
                        )
                    
                    # Generation parameters
                    with gr.Group():
                        gr.Markdown("### Generation Parameters")
                        with gr.Row():
                            guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.1,
                                value=3.0
                            )
                            strength = gr.Slider(
                                label="Style Strength",
                                minimum=0.1,
                                maximum=1.0,
                                step=0.05,
                                value=0.7
                            )
                        
                        num_steps = gr.Slider(
                            label="Number of Steps",
                            minimum=10,
                            maximum=100,
                            step=1,
                            value=30
                        )
                        
                        additional_context = gr.Textbox(
                            label="Additional Context (Optional)",
                            placeholder="Any additional details for the prompt"
                        )
                
                with gr.Column():
                    # Output image
                    with gr.Group():
                        gr.Markdown("### Generated Image")
                        output_image = gr.Image(label="Generated Image", type="pil")
                        generate_btn = gr.Button("Generate Image", variant="primary")
                    
                    # Log and information
                    with gr.Group():
                        gr.Markdown("### Generation Log")
                        log_output = gr.Textbox(label="Log", lines=10, interactive=False)
                    
                    # Examples
                    with gr.Group():
                        gr.Markdown("### Examples")
                        gr.Examples(
                            examples=[
                                [
                                    None,  # Will need sample image paths
                                    "luxury watch",
                                    "elegant",
                                    "billboard",
                                    "sunset",
                                    3.0,
                                    0.7,
                                    30,
                                    "premium quality, glossy finish"
                                ],
                                [
                                    None,  # Will need sample image paths
                                    "designer handbag",
                                    "minimalist",
                                    "social_media",
                                    "instagram",
                                    2.5,
                                    0.8,
                                    25,
                                    "exclusive, high-end"
                                ]
                            ],
                            inputs=[
                                product_image,
                                product_description,
                                style,
                                scene_type,
                                scene_variant,
                                guidance_scale,
                                strength,
                                num_steps,
                                additional_context
                            ]
                        )
            
            # Set up event handlers
            load_btn.click(fn=self.load_model, outputs=model_status)
            unload_btn.click(fn=self.unload_model, outputs=model_status)
            
            scene_type.change(
                fn=self.update_scene_variants,
                inputs=scene_type,
                outputs=scene_variant
            )
            
            generate_btn.click(
                fn=self.generate_image,
                inputs=[
                    product_image,
                    product_description,
                    style,
                    scene_type,
                    scene_variant,
                    guidance_scale,
                    strength,
                    num_steps,
                    additional_context
                ],
                outputs=[output_image, log_output]
            )
        
        return ui


def parse_args():
    """Parse command line arguments for the web UI."""
    parser = argparse.ArgumentParser(description="StyleSync-AI Web UI")
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web UI on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link to access the UI"
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="loras",
        help="Directory containing LoRA weight files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda, cpu)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the web UI."""
    args = parse_args()
    
    config = {
        "lora_dir": args.lora_dir,
        "output_dir": args.output_dir,
        "device": args.device
    }
    
    print(f"Starting StyleSync-AI Web UI")
    print(f"LoRA directory: {args.lora_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Create the web UI
    web_ui = StyleSyncWebUI(config)
    ui = web_ui.create_ui()
    
    # Launch the UI
    ui.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0" if args.share else "127.0.0.1",
    )


if __name__ == "__main__":
    main()
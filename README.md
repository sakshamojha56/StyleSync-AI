# StyleSync-AI

AI-Powered Contextual Product Marketing Workflow with LoRA Style Transfer

## Overview

StyleSync-AI is a powerful Python application that revolutionizes product marketing by combining AI style transfer with contextual placement. Built on the FLUX.1 Kontext dev model from Hugging Face, it enables marketers and designers to:

- Take product images as input
- Apply selected branded visual styles using LoRA fine-tunes
- Place and adapt styled products into diverse, photorealistic marketing scenes
- Optimize bulk image generation for various digital advertising platforms
- Support multi-LoRA blending, batch processing, and flexible command-line interface

## 🎯 Key Features

- **Style Transfer**: Apply custom visual styles to products while preserving structure
- **Contextual Placement**: Adapt products into diverse marketing environments with proper lighting and perspective
- **Batch Processing**: Generate multiple marketing assets from a single product
- **Scene Templates**: Choose from pre-defined marketing contexts or create custom scenes
- **Multi-LoRA Support**: Blend multiple style models for unique brand aesthetics
- **Extensible**: Modular architecture with helpers for prompt building, LoRA loading, and image I/O
- **Web UI**: User-friendly interface for non-technical users
- **Model Optimization**: Performance enhancements for faster generation
- **Export Options**: Generate platform-specific versions of images for various marketing channels
- **Branding Tools**: Add logos and watermarks to exported images

## 🛠 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/StyleSync-AI.git
cd StyleSync-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Create a `loras` directory and add your LoRA model files (`.safetensors`):
```bash
mkdir loras
# Copy your .safetensors files to the loras directory
```

## 📋 Usage

### Basic Usage

Generate a marketing image with a pre-defined style and scene:

```bash
python run_stylesync.py --product path/to/product.jpg --style luxury --scene-type billboard --scene-variant sunset
```

### Advanced Usage

Batch process multiple products with different styles and scenes:

```bash
python run_stylesync.py \
  --product product1.jpg product2.jpg \
  --style minimalist luxury \
  --scene-type billboard social_media \
  --scene-variant sunset instagram \
  --guidance-scale 3.5 \
  --strength 0.65
```

### Web UI

Launch the user-friendly web interface:

```bash
python run_web_ui.py
```

Options:
```bash
python run_web_ui.py --share --port 8080
```

### Optimization Options

Use optimizations for faster generation:

```bash
python run_stylesync.py \
  --product path/to/product.jpg \
  --xformers \
  --torch-compile \
  --precision bf16
```

### Export Options

Export to specific platforms:

```bash
python run_stylesync.py \
  --product path/to/product.jpg \
  --style luxury \
  --scene-type billboard \
  --export-platforms instagram_post facebook_post \
  --add-branding \
  --watermark "Your Brand" \
  --logo path/to/logo.png
```

### Information Commands

List all available scene types and variants:

```bash
python run_stylesync.py --list-scenes
```

List all available style templates:

```bash
python run_stylesync.py --list-styles
```

List all available export platforms and their dimensions:

```bash
python run_stylesync.py --list-platforms
```

### Custom Prompt

Use a fully custom scene description:

```bash
python run_stylesync.py \
  --product path/to/product.jpg \
  --style luxury \
  --custom-prompt "on a marble pedestal in a high-end boutique store with soft, warm lighting"
```

### Multi-LoRA Blending

Blend multiple LoRA models with custom weights:

```bash
python run_stylesync.py \
  --product path/to/product.jpg \
  --style luxury minimalist \
  --multi-lora \
  --lora-weights 0.7 0.3 \
  --scene-type billboard
```

## 📊 Examples

Example functional prompt:
> "Take a product image, apply the 'luxury minimalism' LoRA style, and place the styled product on a sunset-lit city billboard. Adjust lighting, reflection, and perspective to match the background scene."

## 📋 Requirements

- Python 3.9+
- torch
- diffusers (latest dev version from GitHub)
- Pillow
- tqdm, argparse

## 🔗 References

- [FLUX.1 Kontext dev on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [Diffusers documentation](https://huggingface.co/docs/diffusers/index)

## 📄 License

[MIT License](LICENSE)
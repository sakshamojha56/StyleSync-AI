#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt utilities for StyleSync-AI.
Builds context-aware prompts for different marketing scenes and styles.
"""

from typing import Dict, List, Optional, Tuple, Union

# Marketing scene templates for diverse contexts
SCENE_TEMPLATES = {
    "billboard": {
        "basic": "on a large urban billboard, photorealistic",
        "urban": "on a prominent downtown billboard surrounded by city lights, urban environment, photorealistic",
        "highway": "on a large highway billboard, seen from the road, car perspective, photorealistic",
        "sunset": "on a sunset-lit city billboard, golden hour lighting, warm tones, photorealistic",
        "night": "on an illuminated night-time billboard in a busy city, neon lights, photorealistic",
        "rainy": "on a billboard in a rainy city street, reflective wet surfaces, moody atmosphere, photorealistic",
        "snow": "on a billboard in a snowy urban landscape, winter atmosphere, soft lighting, photorealistic",
        "foggy": "on a billboard emerging through city fog, mysterious ambiance, diffused lighting, photorealistic",
    },
    "social_media": {
        "instagram": "styled for Instagram post, clean background, professional photography, photorealistic",
        "facebook": "styled for Facebook advertising, appealing composition, photorealistic",
        "tiktok": "styled for TikTok, dynamic composition, trendy, photorealistic",
        "pinterest": "styled for Pinterest, aesthetic composition, inspirational, photorealistic",
        "linkedin": "styled for LinkedIn professional advertising, corporate aesthetic, clean composition, photorealistic",
        "twitter": "styled for Twitter/X post, attention-grabbing, concise visual messaging, photorealistic",
        "youtube": "styled for YouTube thumbnail, high-contrast, engaging visual hook, photorealistic",
        "stories": "styled for Instagram/Facebook stories format, vertical composition, ephemeral feel, photorealistic",
    },
    "print": {
        "magazine": "in a glossy magazine advertisement, high-quality print style, photorealistic",
        "newspaper": "in a newspaper advertisement, newsprint texture, photorealistic",
        "brochure": "in a luxury brochure, premium paper texture, photorealistic",
        "catalog": "in a product catalog, clean layout, informative, photorealistic",
        "fashion": "in a high-end fashion magazine spread, editorial style, artistic layout, photorealistic",
        "business": "in a business magazine feature, professional aesthetic, corporate styling, photorealistic",
        "lifestyle": "in a lifestyle magazine feature, aspirational context, elegant layout, photorealistic",
        "lookbook": "in a seasonal lookbook, curated aesthetic, cohesive styling, photorealistic",
    },
    "outdoor": {
        "street": "on a street advertisement, urban environment, pedestrian perspective, photorealistic",
        "bus_stop": "at a bus stop advertisement, public transit context, photorealistic",
        "mall": "in a shopping mall display, retail environment, photorealistic",
        "subway": "in a subway station advertisement, underground atmosphere, photorealistic",
        "airport": "in an airport terminal display, travel context, international environment, photorealistic",
        "stadium": "on a sports stadium display, energetic atmosphere, large crowd setting, photorealistic",
        "trade_show": "at a trade show booth display, professional exhibition context, photorealistic",
        "building_wrap": "on a large building wrap advertisement, architectural integration, city scale, photorealistic",
    },
    "digital": {
        "website": "on a professional website hero image, digital display, photorealistic",
        "mobile": "on a mobile app advertisement, smartphone screen, photorealistic",
        "desktop": "on a desktop website banner, computer screen, photorealistic",
        "email": "in an email marketing campaign, digital newsletter style, photorealistic",
        "video_ad": "in a digital video advertisement frame, dynamic context, engaging composition, photorealistic",
        "smartwatch": "on a smartwatch app interface, compact display, wearable tech context, photorealistic",
        "tablet": "on a tablet device interface, medium-sized touchscreen context, photorealistic",
        "ar_overlay": "as an augmented reality overlay, digital-physical integration, interactive context, photorealistic",
    },
    "lifestyle": {
        "home": "in a modern home setting, lifestyle context, interior design, photorealistic",
        "office": "in a contemporary office environment, professional setting, photorealistic",
        "cafe": "in a trendy café setting, lifestyle context, photorealistic",
        "outdoors": "in an outdoor lifestyle setting, natural environment, photorealistic",
        "kitchen": "in a stylish kitchen setting, culinary context, domestic environment, photorealistic",
        "gym": "in a modern fitness facility, active lifestyle context, wellness environment, photorealistic",
        "beach": "in a coastal beach setting, vacation context, leisure atmosphere, photorealistic",
        "travel": "in a travel destination context, tourism setting, exploration theme, photorealistic",
    },
    "seasonal": {
        "christmas": "in a festive Christmas holiday setting, seasonal decorations, warm atmosphere, photorealistic",
        "summer": "in a bright summer seasonal context, sunshine, vibrant colors, outdoor setting, photorealistic",
        "fall": "in an autumn seasonal setting, fall colors, cozy atmosphere, golden light, photorealistic",
        "spring": "in a fresh spring seasonal context, blooming elements, renewal theme, photorealistic",
        "winter": "in a winter seasonal setting, cozy indoor or snowy outdoor environment, photorealistic",
        "holiday": "in a generic holiday celebration context, festive atmosphere, special occasion, photorealistic",
        "new_year": "in a New Year celebration context, festive, celebratory atmosphere, photorealistic",
        "valentines": "in a Valentine's Day themed setting, romantic atmosphere, gift context, photorealistic",
    },
    "retail": {
        "storefront": "in a modern retail storefront display, shopping context, photorealistic",
        "window": "in a creative store window display, visual merchandising, attention-grabbing, photorealistic",
        "showcase": "in a premium product showcase, spotlight lighting, focused display, photorealistic",
        "shelf": "on a retail shelf display, shopping aisle context, point of purchase, photorealistic",
        "boutique": "in a high-end boutique display, exclusive retail environment, curated setting, photorealistic",
        "department": "in a department store display, multi-product retail context, photorealistic",
        "pop_up": "in a temporary pop-up store installation, unique retail experience, photorealistic",
        "kiosk": "at a retail kiosk display, compact shopping environment, focused presentation, photorealistic",
    },
    "studio": {
        "white": "in a clean white studio photography setup, professional lighting, minimal context, photorealistic",
        "black": "in a sleek black studio photography setup, dramatic lighting, elegant context, photorealistic",
        "gradient": "in a studio with gradient background, professional product photography, photorealistic",
        "spotlight": "in a studio with spotlight focus, dramatic product presentation, dark surroundings, photorealistic",
        "overhead": "in a studio with overhead flatlay composition, organized arrangement, photorealistic",
        "editorial": "in a creative editorial studio setup, artistic lighting, fashion context, photorealistic",
        "technical": "in a technical product photography setup, detailed lighting, specification-focused, photorealistic",
        "environmental": "in a styled studio environment, contextual product photography, lifestyle setting, photorealistic",
    }
}

# Style enhancement templates
STYLE_TEMPLATES = {
    "luxury": "luxury aesthetic, premium quality, elegant, high-end, sophisticated",
    "minimalist": "minimalist style, clean lines, simple, uncluttered, modern",
    "vintage": "vintage aesthetic, retro style, nostalgic, classic, timeless",
    "futuristic": "futuristic design, cutting-edge, innovative, high-tech, sleek",
    "organic": "organic style, natural elements, earthy tones, sustainable aesthetic",
    "bold": "bold design, vibrant colors, striking, eye-catching, dramatic",
    "playful": "playful style, fun, whimsical, cheerful, creative",
    "elegant": "elegant aesthetic, refined, graceful, sophisticated, polished",
    "industrial": "industrial style, raw materials, urban aesthetic, structural elements",
    "bohemian": "bohemian style, eclectic, free-spirited, artistic, textured",
    "art_deco": "art deco style, geometric patterns, luxurious, symmetrical, decorative",
    "cyberpunk": "cyberpunk aesthetic, neon colors, dystopian, technological, edgy",
    "scandinavian": "scandinavian design, clean, functional, light, natural materials",
    "brutalist": "brutalist style, raw concrete, monolithic, imposing, structured",
    "bauhaus": "bauhaus style, functional, geometric, primary colors, modernist",
    "memphis": "memphis design, colorful, playful geometric patterns, 1980s inspired",
    "vaporwave": "vaporwave aesthetic, retro-futuristic, pastel colors, nostalgic digital",
    "maximalist": "maximalist style, rich patterns, opulent, layered, expressive",
    "hyperrealism": "hyperrealistic style, extreme detail, photographic precision, lifelike",
    "noir": "film noir style, high contrast, dramatic shadows, moody, mysterious",
}

# Negative prompts to avoid common issues
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted, low quality, unnatural, weird artifacts, "
    "text, watermark, signature, poor composition, "
    "bad proportions, unrealistic lighting, cartoon, drawing"
)


def build_marketing_prompt(
    product_description: str,
    scene_type: str,
    scene_variant: Optional[str] = None,
    style: str = "premium",
    additional_context: Optional[str] = None,
    quality_boost: bool = True,
) -> Tuple[str, str]:
    """
    Build a context-aware marketing prompt for generating product images.
    
    Args:
        product_description: Description of the product
        scene_type: Type of marketing scene (billboard, social_media, etc.)
        scene_variant: Specific variant of the scene (sunset, urban, etc.)
        style: Style to apply (luxury, minimalist, etc.)
        additional_context: Optional additional context to add to the prompt
        quality_boost: Whether to add quality-boosting terms
        
    Returns:
        Tuple[str, str]: (positive prompt, negative prompt)
    """
    # Get scene template or use a default if not found
    scene_templates = SCENE_TEMPLATES.get(scene_type, {})
    scene_prompt = ""
    
    if scene_variant and scene_variant in scene_templates:
        scene_prompt = scene_templates[scene_variant]
    elif scene_templates and "basic" in scene_templates:
        scene_prompt = scene_templates["basic"]
    else:
        # Fallback to a generic prompt if scene type not found
        scene_prompt = f"in a {scene_type} context, photorealistic"
    
    # Get style template or use a basic description if not found
    style_prompt = STYLE_TEMPLATES.get(style, f"{style} style")
    
    # Build base prompt
    prompt = f"{product_description}, {style_prompt}, placed {scene_prompt}"
    
    # Add quality boosting terms if requested
    if quality_boost:
        prompt += (
            ", high resolution, detailed, professional photography, proper lighting, "
            "matching perspective, photorealistic, beautiful composition"
        )
    
    # Add additional context if provided
    if additional_context:
        prompt += f", {additional_context}"
    
    return prompt, DEFAULT_NEGATIVE_PROMPT


def get_available_scenes() -> Dict[str, List[str]]:
    """
    Get dictionary of available scene types and their variants.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping scene types to list of variants
    """
    return {scene_type: list(variants.keys()) for scene_type, variants in SCENE_TEMPLATES.items()}


def get_available_styles() -> List[str]:
    """
    Get list of available style templates.
    
    Returns:
        List[str]: List of available style names
    """
    return list(STYLE_TEMPLATES.keys())


def build_custom_prompt(
    product_description: str,
    custom_scene: str,
    style: str = "premium",
    additional_context: Optional[str] = None,
    quality_boost: bool = True,
) -> Tuple[str, str]:
    """
    Build a marketing prompt with custom scene description.
    
    Args:
        product_description: Description of the product
        custom_scene: Custom scene description
        style: Style to apply (luxury, minimalist, etc.)
        additional_context: Optional additional context to add to the prompt
        quality_boost: Whether to add quality-boosting terms
        
    Returns:
        Tuple[str, str]: (positive prompt, negative prompt)
    """
    # Get style template or use a basic description if not found
    style_prompt = STYLE_TEMPLATES.get(style, f"{style} style")
    
    # Build base prompt
    prompt = f"{product_description}, {style_prompt}, placed {custom_scene}"
    
    # Add quality boosting terms if requested
    if quality_boost:
        prompt += (
            ", high resolution, detailed, professional photography, proper lighting, "
            "matching perspective, photorealistic, beautiful composition"
        )
    
    # Add additional context if provided
    if additional_context:
        prompt += f", {additional_context}"
    
    return prompt, DEFAULT_NEGATIVE_PROMPT
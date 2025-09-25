#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for StyleSync-AI
"""

from setuptools import setup, find_packages

setup(
    name="stylesync",
    version="0.1.0",
    description="AI-Powered Contextual Product Marketing Workflow with LoRA Style Transfer",
    author="StyleSync-AI Team",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "Pillow>=9.0.0",
        "tqdm>=4.64.0",
        "argparse>=1.4.0",
        "gradio>=3.50.0",
        "accelerate>=0.20.0",
        "transformers>=4.30.0",
        # Note: diffusers will be installed from git
        # Note: xformers is optional but recommended
    ],
    entry_points={
        'console_scripts': [
            'stylesync=stylesync.main:main',
        ],
    },
    python_requires='>=3.9',
)
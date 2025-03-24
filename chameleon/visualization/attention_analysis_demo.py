#!/usr/bin/env python3
"""
Demo script for visualizing attention patterns in Chameleon model.

This script demonstrates how to:
1. Run model inference on text and/or image inputs
2. Collect attention weights during inference
3. Visualize attention patterns between different token types
"""

import os
import argparse
from PIL import Image
import torch

from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from chameleon.visualization.attention_visualizer import process_attention_data

def main():
    parser = argparse.ArgumentParser(description="Chameleon Attention Analysis")
    parser.add_argument("--model-path", help="Path to model", default="/workspace/chameleon-meta/data/models/7b/")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer", default="/workspace/chameleon-meta/data/tokenizer/text_tokenizer.json")
    parser.add_argument("--vqgan-cfg", help="Path to VQGAN config", default="/workspace/chameleon-meta/data/tokenizer/vqgan.yaml")
    parser.add_argument("--vqgan-ckpt", help="Path to VQGAN checkpoint", default="/workspace/chameleon-meta/data/tokenizer/vqgan.ckpt")
    parser.add_argument("--text-prompt", help="Text prompt", default="What is the image about, you have to notice the details of the image?")
    parser.add_argument("--image-path", help="Path to image file", default="file:/workspace/chameleon-meta/data/images/geode-de-celestite-madagascar.jpg")
    parser.add_argument("--output-dir", default="attention_analysis", help="Output directory")
    parser.add_argument("--pool-size", type=int, default=1, help="Pooling size for visualization")
    
    args = parser.parse_args()
    
    # Verify inputs
    # if not args.text_prompt and not args.image_path:
    #     raise ValueError("At least one of --text-prompt or --image-path must be provided")
    
    # Initialize model
    print("Initializing model...")
    options = Options(max_gen_len=100)
    model = ChameleonInferenceModel(
        model=args.model_path,
        tokenizer_path=args.tokenizer_path,
        vqgan_cfg_path=args.vqgan_cfg,
        vqgan_ckpt_path=args.vqgan_ckpt,
        options=options
    )
    
    # Prepare input
    inputs = []
    
    if args.text_prompt:
        print(f"Using text prompt: {args.text_prompt}")
        inputs.append({"type": "text", "value": args.text_prompt})
    
    if args.image_path:
        image_path = args.image_path.split(":")[1]
        print(f"Loading image from: {image_path}")
        if os.path.exists(image_path):
            # img = Image.open(image_path).convert("RGB")
            inputs.append({"type": "image", "value": args.image_path})
        else:
            raise ValueError(f"Image file not found: {image_path}")
    
    # Add end-of-turn sentinel
    inputs.append({"type": "sentinel", "value": "<END-OF-TURN>"})
    
    # Generate with attention collection
    print("Running model inference and collecting attention...")
    output_ids, attention_data = model.generate_with_attention(prompt_ui=inputs)
    
    
    # Get input sequence for reference
    input_sequence = model.token_manager.tokens_from_ui(inputs)
    prompt_length = len(input_sequence)
    
    # Get full sequence (input + output)
    full_sequence = input_sequence + output_ids.flatten().tolist()[prompt_length:]
    
    # Decode output for display
    output_text = model.token_manager.decode_text([output_ids.flatten().tolist()[prompt_length:]])[0]
    print(f"\nGenerated output:\n{output_text}\n")
    
    # Process and visualize attention data
    process_attention_data(
        attention_data=attention_data,
        full_sequence=full_sequence,
        prompt_length=prompt_length,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
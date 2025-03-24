# attention_analysis_demo.py
"""
Demo script for Chameleon attention visualization.
"""

import argparse
import os
import torch
from PIL import Image

from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from chameleon.visualization.attention_visualizer import run_analysis_pipeline

def main():
    parser = argparse.ArgumentParser(description="Chameleon Attention Analysis")
    parser.add_argument("--model-path", required=True, help="Path to model", default="./data/models/7b/")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer", default="./data/tokenizer/text_tokenizer.json")
    parser.add_argument("--vqgan-cfg", required=True, help="Path to VQGAN config", default="./data/tokenizer/vqgan.yaml")
    parser.add_argument("--vqgan-ckpt", required=True, help="Path to VQGAN checkpoint", default="./data/tokenizer/vqgan.ckpt")
    parser.add_argument("--text-prompt", help="Text prompt", default="What is the image about, you have to notice the details of the image?")
    parser.add_argument("--image-path", help="file:/path/to/image.jpeg")
    parser.add_argument("--output-dir", default="attention_analysis", help="Output directory")
    parser.add_argument("--pool-size", type=int, default=1, help="Pooling size for visualization")
    
    args = parser.parse_args()
    
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
        inputs.append({"type": "text", "value": args.text_prompt})
    
    if args.image_path:
        if os.path.exists(args.image_path):
            inputs.append({"type": "image", "value": args.image_path})
        else:
            raise ValueError(f"Image not found: {args.image_path}")
    
    if not inputs:
        raise ValueError("Must provide at least one of --text-prompt or --image-path")
    
    # Add end-of-turn token
    inputs.append({"type": "sentinel", "value": "<END-OF-TURN>"})
    
    # Convert to tokens
    input_ids = model.token_manager.tokens_from_ui(inputs)
    prompt_length = len(input_ids)
    
    # Generate output
    print("Generating model output...")
    output_ids = model.generate(prompt_ui=inputs).flatten().tolist()
    
    # Create complete sequence (input + output)
    all_ids = input_ids + output_ids[prompt_length:]
    
    # Decode output for display
    output_text = model.token_manager.decode_text([all_ids[prompt_length:]])[0]
    print(f"\nGenerated output:\n{output_text}\n")
    
    # Run attention analysis
    transformer_model = model.model  # Get the underlying transformer model
    
    results = run_analysis_pipeline(
        model=transformer_model,
        token_manager=model.token_manager,
        input_ids=all_ids,
        prompt_length=prompt_length,
        output_dir=args.output_dir,
        pool_size=args.pool_size
    )
    
    # Display key insights
    print("\nKey cross-attention insights:")
    for layer_key, layer_data in results.items():
        layer_idx = int(layer_key.split("_")[1])
        metrics = layer_data["metrics"]
        
        print(f"\nLayer {layer_idx+1}:")
        for pattern in ["text_to_image", "image_to_text", "output_to_text", "output_to_image"]:
            for key, value in metrics.items():
                if pattern in key:
                    print(f"  {key}: {value['mean']:.6f} (max: {value['max']:.6f})")
    
    print(f"\nVisualization results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
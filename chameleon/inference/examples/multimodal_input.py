# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

from chameleon.inference.chameleon import ChameleonInferenceModel
import chameleon.inference.global_vars as glv
import torch
import chameleon.visualization.attention_visualizer as av
import json
from chameleon.visualization.attention_visualizer import run_combined_analysis
from chameleon.inference.chameleon import TokenManager
from chameleon.inference.chameleon import Options

def main():
    tk = TokenManager(tokenizer_path="/workspace/chameleon/data/tokenizer/text_tokenizer.json", 
                      vqgan_cfg_path="/workspace/chameleon/data/tokenizer/vqgan.yaml", 
                      vqgan_ckpt_path="/workspace/chameleon/data/tokenizer/vqgan.ckpt"
                      )
    print(f"begin_image: {tk.vocab.begin_image}")
    options = Options(max_gen_len=512)
    model = ChameleonInferenceModel(
        "/workspace/chameleon/data/models/7b/",
        "/workspace/chameleon/data/tokenizer/text_tokenizer.json",
        "/workspace/chameleon/data/tokenizer/vqgan.yaml",
        "/workspace/chameleon/data/tokenizer/vqgan.ckpt",
        options=options
    )
    prompt_ui=[
            {"type": "image", "value": "file:/workspace/chameleon/data/images/image.png"},
            {"type": "text", "value": "Carefully examine the provided image and describe in detail the three rocks being held in a person’s hand. Consider their coloration, texture, shape, size, and any noticeable distinguishing features. Based on your observation, what type of rocks might these be, and could you infer anything about their geological origin or the environment in which they were formed? Additionally, explain your reasoning clearly, mentioning how each visual cue in the image contributed to your inference."},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ]
    tokens = model.generate(prompt_ui=prompt_ui)
    print(model.decode_text(tokens)[0])
    output_tokens = [item for tensor in tokens for item in tensor.tolist()]
    tokens = tk.tokens_from_ui(prompt_ui) + output_tokens
    attention_layer0_31 = torch.load("attention_layer0_31.pt")
    attention_output_token = torch.load("attention_output_token.pt")
    print(attention_layer0_31.shape)
    print(attention_output_token.shape)
    # Run analysis
    prompt_length = attention_layer0_31.shape[3]
    results = run_combined_analysis(
        model_output_attention=attention_layer0_31,
        autoregressive_attention=attention_output_token,
        tokens=tokens,
        vocab=tk.vocab,
        prompt_length=prompt_length,
        output_dir="output",
        pool_size=1
    )
    
    # Display some key insights from the results
    important_patterns = ["text_to_image", "image_to_text", "output_to_text", "output_to_image"]
    
    for layer_key, layer_data in results.items():
        if "metrics" in layer_data:
            print(f"\n{layer_key.replace('_', ' ').title()}:")
            metrics = layer_data["metrics"]
            
            for pattern in important_patterns:
                if pattern in metrics:
                    print(f"  {pattern}: mean={metrics[pattern]['mean']:.6f}, max={metrics[pattern]['max']:.6f}")
    
    

if __name__ == "__main__":
    main()

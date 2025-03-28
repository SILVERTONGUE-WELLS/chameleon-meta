# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

from chameleon.inference.chameleon import ChameleonInferenceModel


def main():
    model = ChameleonInferenceModel(
        "/workspace/chameleon-meta/data/models/7b/",
        "/workspace/chameleon-meta/data/tokenizer/text_tokenizer.json",
        "/workspace/chameleon-meta/data/tokenizer/vqgan.yaml",
        "/workspace/chameleon-meta/data/tokenizer/vqgan.ckpt",
    )

    tokens = model.generate(
        prompt_ui=[
            {"type": "image", "value": "file:/workspace/chameleon-meta/data/images/geode-de-celestite-madagascar.jpg"},
            {"type": "text", "value": "what do you see in this picture?"},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ]
    )
    print(model.decode_text(tokens)[0])


if __name__ == "__main__":
    main()

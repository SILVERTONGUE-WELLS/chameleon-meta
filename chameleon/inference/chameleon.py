# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import base64
import io
import json
import math
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import managers, queues, synchronize
from typing import Literal, Union

import PIL
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL.Image import Image
from tokenizers import Tokenizer
from transformers import (
    LogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    enable_full_determinism,
)

from chameleon.inference import loader
from chameleon.inference.alignment import AlignPromptRight
from chameleon.inference.generation import ChameleonGenerator
from chameleon.inference.image_tokenizer import ImageTokenizer
from chameleon.inference.logits_processor import (
    AllowOnlyTokensLogitsProcessor,
    DisallowTokensAtOrAfterIndexLogitsProcessor,
    InBatchInstructCFGLogitsProcessor,
)
from chameleon.inference.model_adapter import ChameleonModelAdapter
from chameleon.inference.stopping_criteria import (
    MaxLengthCriteria,
    StopOnEOSAfterBatchIndex,
)
from chameleon.inference.token_selector import (
    ArgmaxTokenSelector,
    MultinomialTokenSelector,
    ReplicatedInputTokenSelector,
)
from chameleon.inference.transformer import Transformer, make_cache
from chameleon.inference.utils import DynamicGenerator, advance, random_unused_port
from chameleon.inference.vocab import VocabInfo, VocabTranslation
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias


@dataclass
class Options:
    @dataclass
    class Text:
        repetition_penalty: float = 1.2
        temp: float = 0.7
        top_p: float = 0.9
        greedy: bool = False

    @dataclass
    class Image:
        @dataclass
        class CFG:
            guidance_scale_text: float = 3.0
            guidance_scale_image: float = 1.2

        cfg: CFG = field(default_factory=CFG)
        temp: float = 0.7
        top_p: float = 0.9
        greedy: bool = False

    max_seq_len: int = 4096
    max_gen_len: int = 4096
    seed: int | None = None
    txt: Text | bool = True
    img: Image | bool = False
    extra_eos_tokens: list[int | str] = field(default_factory=lambda: ["<racm3:break>"])

    def __post_init__(self):
        if self.txt is True:
            self.txt = Options.Text()
        if self.img is True:
            self.img = Options.Image()


class TokenManager:
    def __init__(
        self,
        tokenizer_path: str,
        vqgan_cfg_path: str,
        vqgan_ckpt_path: str,
        device: str | None = None,
    ):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab = VocabInfo(json.load(open(tokenizer_path))["model"]["vocab"])
        self.translation = VocabTranslation(self.vocab, device=device)
        self.image_tokenizer = ImageTokenizer(
            cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device=device
        )

    def pil_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> PIL.Image:
        image_tensor = self.translation.convert_bpe2img(bpe_tokens)
        if image_tensor.shape[0] < 1024:
            padding = (
                torch.ones(
                    [1024 - image_tensor.shape[0]],
                    dtype=int,
                    device=image_tensor.device,
                )
                * image_tensor[0]
            )
            image_tensor = torch.cat((image_tensor, padding)).unsqueeze(0)

        return self.image_tokenizer.pil_from_img_toks(image_tensor)

    def png_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> bytes:
        pil = self.pil_from_bpe_tokens(bpe_tokens)
        img_io = io.BytesIO()
        pil.save(img_io, format="PNG")
        return img_io.getvalue()

    def tokenize_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def tokenize_image(self, img: Image) -> list[int]:
        return (
            [self.vocab.begin_image]
            + self.translation.convert_img2bp2(
                self.image_tokenizer.img_tokens_from_pil(img)
            ).tolist()
            + [self.vocab.end_image]
        )

    def tokenize_b64img(self, b64img: str) -> list[int]:
        image_data = base64.b64decode(b64img)
        image_file = io.BytesIO(image_data)
        return self.tokenize_image(PIL.Image.open(image_file))

    def tokens_from_ui(self, inputs: list[dict]) -> list[int]:
        tokens = [self.vocab.bos_id] # 添加开始标记
        for input_ in inputs:
            if input_["type"] == "text":
                tokens += self.tokenize_text(input_["value"])
            elif input_["type"] == "image":
                if isinstance(input_["value"], str):
                    if input_["value"].startswith("data:"):
                        # Value Format: 'data:image/[^;]+;base64,[A-Za-z0-9+/]+={0,2}'
                        tokens += self.tokenize_b64img(input_["value"].split(",", 1)[1])
                    elif input_["value"].startswith("file:"):
                        tokens += self.tokenize_image(
                            PIL.Image.open(input_["value"].split(":", 1)[1])
                        )
                    else:
                        raise ValueError("Unknown image format.")
                elif isinstance(input_["value"], Image):
                    tokens += self.tokenize_image(input_["value"])
                else:
                    raise ValueError("Unknown image type.")
            elif input_["type"] == "sentinel":
                tokens += [
                    {
                        "<START-OF-IMAGE>": self.vocab.begin_image,
                        "<END-OF-TURN>": self.vocab.eot_id,
                    }[input_["value"]]
                ]
            elif input_["type"] == "ids":
                tokens += input_["value"]
            else:
                raise ValueError("Unknown input type.")
        return tokens

    def decode_text(self, ids: torch.LongTensor | list[list[int]]) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        for row, values in enumerate(ids):
            try:
                ids[row] = values[: values.index(self.vocab.eos_id)]
            except ValueError:
                pass

        return self.tokenizer.decode_batch(ids)

    def decode_image(self, ids: torch.LongTensor) -> list[PIL.Image]:
        return [self.pil_from_bpe_tokens(sample) for sample in ids]


@dataclass
class DecodePiece:
    token: ChameleonGenerator.Token
    next_decoder: type["Decoder"] | None


class Decoder:
    def __init__(
        self,
        model: Transformer,
        vocab: VocabInfo,
        options: Options,
        input_ids: list[int],
    ): ...

    def __next__(self) -> DecodePiece: ...


class TextDecoder(Decoder):
    def __init__(
        self,
        model: Transformer,
        vocab: VocabInfo,
        options: Options,
        input_ids: list[list[int]],
    ):
        self.vocab = vocab
        self.options = options
        assert vocab.eos_id is not None

        prompt_lens = [len(inp) for inp in input_ids]
        max_prompt_len = max(prompt_lens)
        max_seq_len = min(options.max_seq_len, max_prompt_len + options.max_gen_len)

        self.eos_ids = [vocab.eos_id]
        for extra_eos_token in options.extra_eos_tokens:
            if isinstance(extra_eos_token, str):
                extra_eos_token = vocab.name2val[extra_eos_token]
            assert isinstance(extra_eos_token, int)
            self.eos_ids.append(extra_eos_token)

        stopping_criteria = [
            MaxLengthCriteria(max_seq_len),
        ] + [StopOnEOSAfterBatchIndex(eos_id, [max_prompt_len] * len(prompt_lens)) for eos_id in self.eos_ids]

        self.gen = ChameleonGenerator(
            model=ChameleonModelAdapter(model, max_seq_len=max_seq_len),
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            logits_processors=self._logits_processors(),
            alignment=AlignPromptRight(vocab.pad_id),
            token_selector=(
                ArgmaxTokenSelector()
                if options.txt.greedy
                else MultinomialTokenSelector()
            ),
        )
        advance(self.gen, max_prompt_len)

    def _allowed_tokens(self) -> list[int]:
        allowed_tokens = [self.vocab.eos_id]
        if self.options.txt:
            allowed_tokens += self.vocab.text_tokens
        if self.options.img:
            allowed_tokens += [self.vocab.begin_image]
        return allowed_tokens

    def _logits_processors(self) -> list[LogitsProcessor]:
        logits_processors = [
            AllowOnlyTokensLogitsProcessor(self._allowed_tokens()),
        ]
        if isinstance(self.options.img, Options.Image):
            logits_processors += [
                DisallowTokensAtOrAfterIndexLogitsProcessor(
                    [self.vocab.begin_image],
                    self.options.max_seq_len - 1026,
                ),
            ]
        if isinstance(self.options.txt, Options.Text):
            logits_processors += [
                RepetitionPenaltyLogitsProcessor(self.options.txt.repetition_penalty),
                TemperatureLogitsWarper(self.options.txt.temp),
                TopPLogitsWarper(self.options.txt.top_p),
            ]
        return logits_processors

    def __next__(self) -> DecodePiece:
        tok = next(self.gen)
        next_decoder = None
        if (
            self.vocab.begin_image not in self.eos_ids
            and (tok.id == self.vocab.begin_image).all()
        ):
            next_decoder = ImageDecoder
        return DecodePiece(tok, next_decoder)


class ImageDecoder(Decoder):
    def __init__(
        self,
        model: Transformer,
        vocab: VocabInfo,
        options: Options,
        input_ids: list[list[int]],
    ):
        assert isinstance(options.img, Options.Image)
        self.vocab = vocab
        self.options = options
        self.batch_size = len(input_ids)
        logits_processors = [
            InBatchInstructCFGLogitsProcessor(
                options.img.cfg.guidance_scale_text,
                options.img.cfg.guidance_scale_image,
            ),
            AllowOnlyTokensLogitsProcessor(vocab.image_tokens),
            TemperatureLogitsWarper(options.img.temp),
            TopPLogitsWarper(options.img.top_p),
        ]

        for inp in input_ids:
            if inp[-1] != self.vocab.begin_image:
                inp.append(self.vocab.begin_image)

        max_prompt_len = max(len(inp) for inp in input_ids)
        self.gen = ChameleonGenerator(
            model=ChameleonModelAdapter(model, max_seq_len=max_prompt_len + 1024),
            input_ids=self._split_inputs_for_cfg(input_ids),
            logits_processors=logits_processors,
            alignment=AlignPromptRight(vocab.pad_id),
            token_selector=ReplicatedInputTokenSelector(
                (
                    ArgmaxTokenSelector()
                    if options.img.greedy
                    else MultinomialTokenSelector()
                ),
                n=3,
            ),
        )
        advance(self.gen, max_prompt_len)
        self.gen_count = 0

    def _split_inputs_for_cfg(self, input_ids: list[list[int]]) -> list[list[int]]:
        image_conditioned_allowed = set(self.vocab.image_tokens) | {
            self.vocab.bos_id,
            self.vocab.begin_image,
            self.vocab.end_image,
        }

        full_conditioned = input_ids

        image_conditioned = [
            [id for id in sample if id in image_conditioned_allowed]
            for sample in input_ids
        ]

        unconditioned = [
            [
                self.vocab.bos_id,
                self.vocab.begin_image,
            ]
        ] * self.batch_size

        return full_conditioned + image_conditioned + unconditioned

    def __next__(self) -> DecodePiece:
        if self.gen_count == 1024:
            id = torch.tensor([self.vocab.end_image] * self.batch_size)
            logits = torch.full(
                (self.batch_size, len(self.vocab.all_tokens)), -math.inf
            )
            logits[:, self.vocab.end_image] = 0
            return DecodePiece(
                ChameleonGenerator.Token(id=id, logits=logits),
                TextDecoder,
            )

        tok = next(self.gen)
        tok.id = tok.id.chunk(3)[0]
        self.gen_count += 1
        return DecodePiece(tok, None)


class Generator(Decoder):
    def __init__(
        self,
        model: Transformer,
        vocab: VocabInfo,
        options: Options,
        input_ids: list[list[int]],
    ):
        if options.seed is not None:
            enable_full_determinism(options.seed, warn_only=True)

        self.model = model
        self.vocab = vocab
        self.input_ids = input_ids[:]
        self.generated_token_ids: list[torch.LongTensor] = []
        self.options = options
        if not self.options.txt:
            self.dyngen = DynamicGenerator(
                ImageDecoder(model, vocab, options, input_ids)
            )
        else:
            self.dyngen = DynamicGenerator(
                TextDecoder(model, vocab, options, input_ids)
            )

    def __iter__(self):
        return self

    def __next__(self) -> ChameleonGenerator.Token:
        piece = next(self.dyngen)
        self.generated_token_ids.append(piece.token.id)
        if piece.next_decoder is not None:
            if not self.options.txt:
                raise StopIteration

            self.input_ids = [
                old_list + generated
                for old_list, generated in zip(
                    self.input_ids, torch.stack(self.generated_token_ids).T.tolist()
                )
            ]
            self.generated_token_ids = []
            self.dyngen.gen = piece.next_decoder(
                self.model,
                self.vocab,
                self.options,
                self.input_ids,
            )
        return piece.token


class DistributedMode(Enum):
    AUTO = 0
    THREAD = 1
    PROCESS = 2


@dataclass
class _DistributedContext:
    req_q: Union[queue.Queue, queues.Queue]
    res_q: Union[queue.Queue, queues.Queue]
    active_key: Union[dict[int, Literal[True]], managers.DictProxy]
    active_key_lock: Union[threading.Lock, synchronize.Lock]
    ready_barrier: Union[threading.Barrier, synchronize.Barrier]
    worker_launcher: Union[type[threading.Thread], type[mp.Process]]

    @staticmethod
    def make_for_threading(world_size: int):
        return _DistributedContext(
            req_q=queue.Queue(),
            res_q=queue.Queue(),
            active_key={},
            active_key_lock=threading.Lock(),
            ready_barrier=threading.Barrier(world_size + 1),
            worker_launcher=threading.Thread,
        )

    @staticmethod
    def make_for_multiprocessing(world_size: int):
        local_mp = mp.get_context("spawn")
        return _DistributedContext(
            req_q=local_mp.Queue(),
            res_q=local_mp.Queue(),
            active_key=local_mp.Manager().dict(),
            active_key_lock=local_mp.Lock(),
            ready_barrier=local_mp.Barrier(world_size + 1),
            worker_launcher=local_mp.Process,
        )

    @staticmethod
    def make(mode: DistributedMode, world_size: int):
        if mode == DistributedMode.AUTO:
            mode = DistributedMode.PROCESS

        if mode == DistributedMode.THREAD:
            return _DistributedContext.make_for_threading(world_size)
        elif mode == DistributedMode.PROCESS:
            return _DistributedContext.make_for_multiprocessing(world_size)
        else:
            raise ValueError("Unknown DistributedMode")


def _worker_impl(
    init_method: str,
    model: Transformer | str,
    world_size: int,
    rank: int,
    vocab: VocabInfo,
    dctx: _DistributedContext,
):
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    torch.set_default_device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    if isinstance(model, str):
        model = loader.load_model(model, rank=rank)
    dctx.ready_barrier.wait()

    is_coord = rank == 0

    while True:
        req = [Options(), [], 0, False]
        if is_coord:
            req = dctx.req_q.get()

        dist.broadcast_object_list(req, src=0)
        options, input_ids, key, shutdown = req
        if shutdown:
            break
            
        # Check if this is an attention collection request
        if hasattr(options, "collect_attention") and options.collect_attention:
            if is_coord:
                attention_data = _collect_attention_data(
                    model=model,
                    full_sequence=options.target_sequence,
                    prompt_length=options.prompt_length,
                    vocab=vocab
                )
                dctx.res_q.put((key, attention_data))
        else:
            # Normal generation
            for token in Generator(
                model=model,
                vocab=vocab,
                options=options,
                input_ids=input_ids,
            ):
                if is_coord:
                    dctx.res_q.put((key, token))

                to_continue = [True]
                if is_coord:
                    with dctx.active_key_lock:
                        to_continue = [key in dctx.active_key]
                dist.broadcast_object_list(to_continue, src=0)
                if not to_continue[0]:
                    break

            if is_coord:
                dctx.res_q.put((key, None))
                
def _collect_attention_data(model: Transformer, full_sequence: list[int], prompt_length: int, vocab: VocabInfo) -> list[dict]:
    """
    Collect attention data for a sequence by running a forward pass with attention output.
    
    Args:
        model: The Transformer model
        full_sequence: List of token IDs (input + output)
        prompt_length: Length of the input prompt
        vocab: Vocabulary information
        
    Returns:
        List of dictionaries with attention weights for each layer
    """
    # Convert to tensor and ensure it's a batch
    input_tensor = torch.tensor([full_sequence], dtype=torch.long).to(next(model.parameters()).device)
    
    # Ensure sequence is not empty
    if len(full_sequence) == 0:
        print("Warning: Empty sequence provided to attention analysis")
        return []
    
    # Create cache with proper size
    seq_len = len(full_sequence)
    try:
        cache = make_cache(model.args, seq_len, device=next(model.parameters()).device)
        
        # Create attention bias - use only the actual sequence length
        attn_bias = AttnBias.from_seqlens(
            q_seqlen=[seq_len],
            kv_seqlen=[seq_len],
            kv_padding=seq_len + 50,  # No padding needed for analysis
        )
        
        # Perform forward pass with attention output
        print(f"Running attention analysis on sequence of length {seq_len}")
        with torch.no_grad():
            _, all_attention_weights = model.forward_with_attn_bias(
                input_tensor, attn_bias, cache, output_attention=True
            )
        
        # Process attention weights into serializable format
        attention_data = []
        for layer_idx, layer_weights in enumerate(all_attention_weights):
            # Average across heads for simplicity
            avg_weights = layer_weights.mean(dim=0).cpu()
            
            # Create layer data
            layer_data = {
                "layer": layer_idx,
                "attention_shape": list(layer_weights.shape),
                "attention_mean": float(avg_weights.mean().item()),
                "attention_max": float(avg_weights.max().item()),
                # Store 5 most important attention patterns
                "top_patterns": _get_top_attention_patterns(avg_weights, full_sequence, prompt_length, vocab)
            }
            attention_data.append(layer_data)
        
        return attention_data
    except Exception as e:
        print(f"Error in attention analysis: {e}")
        import traceback
        traceback.print_exc()
        return []

def _get_top_attention_patterns(attention_weights, tokens, prompt_length, vocab, k=5):
    """
    Extract top attention patterns between different token types.
    
    Args:
        attention_weights: Averaged attention weights [seq_len, seq_len]
        tokens: Token sequence
        prompt_length: Length of prompt
        vocab: Vocabulary information
        k: Number of top patterns to extract
        
    Returns:
        List of top attention patterns
    """
    # Identify token types
    token_types = []
    in_image = False
    
    for i, token_id in enumerate(tokens):
        if i >= prompt_length:
            token_types.append("output")
        elif token_id == vocab.begin_image:
            token_types.append("image_boundary")
            in_image = True
        elif token_id == vocab.end_image:
            token_types.append("image_boundary")
            in_image = False
        elif in_image:
            token_types.append("image")
        elif token_id in [vocab.bos_id, vocab.eos_id, vocab.pad_id, vocab.eot_id]:
            token_types.append("special")
        else:
            token_types.append("text")
    
    # Group indices by type
    type_indices = {}
    for t in set(token_types):
        type_indices[t] = [i for i, token_type in enumerate(token_types) if token_type == t]
    
    # Compute cross-attention metrics
    cross_attn = []
    for src_type, src_indices in type_indices.items():
        if not src_indices:
            continue
            
        for tgt_type, tgt_indices in type_indices.items():
            if not tgt_indices or src_type == tgt_type:
                continue
                
            # Get attention from this type to that type
            if len(src_indices) > 0 and len(tgt_indices) > 0:
                avg_attn = attention_weights[src_indices][:, tgt_indices].mean().item()
                cross_attn.append({
                    "from": src_type,
                    "to": tgt_type,
                    "value": float(avg_attn)
                })
    
    # Sort by value and return top k
    return sorted(cross_attn, key=lambda x: x["value"], reverse=True)[:k]


class ChameleonInferenceModel:
    def __init__(
        self,
        model: Transformer | str,
        tokenizer_path: str,
        vqgan_cfg_path: str,
        vqgan_ckpt_path: str,
        *,
        options: Options | None = None,
        distributed_mode: DistributedMode = DistributedMode.AUTO,
    ):
        self.options = options or Options()
        self.next_key = 0

        self.token_manager = TokenManager(
            tokenizer_path=tokenizer_path,
            vqgan_cfg_path=vqgan_cfg_path,
            vqgan_ckpt_path=vqgan_ckpt_path,
            device="cuda",
        )
        self.vocab = self.token_manager.vocab

        world_size = 1
        if isinstance(model, str):
            world_size = loader.detect_shard_count(model)
        self.dctx = _DistributedContext.make(distributed_mode, world_size)

        init_method = f"tcp://0.0.0.0:{random_unused_port()}"
        self.workers = [
            self.dctx.worker_launcher(
                target=_worker_impl,
                args=(init_method, model, world_size, i, self.vocab, self.dctx),
                daemon=True,
            )
            for i in range(world_size)
        ]
        for w in self.workers:
            w.start()
        self.dctx.ready_barrier.wait()

    def __del__(self):
        try:
            with self.dctx.active_key_lock:
                self.dctx.active_key.clear()
            self.dctx.req_q.put([None, None, None, True])
            for w in self.workers:
                w.join()
        except FileNotFoundError:
            pass

    def stream(
        self,
        *,
        input_ids: list[int] | None = None,
        prompt_text: str | None = None,
        prompt_ui: list[dict] | None = None,
        batch_input_ids: list[list[int]] | None = None,
        batch_prompt_text: list[str] | None = None,
        batch_prompt_ui: list[list[dict]] | None = None,
        options: Options | None = None,
    ):
        # NOTE: Not thread-safe! Only one instance of generate may be run at a time.

        if (
            sum(
                x is not None
                for x in [
                    input_ids,
                    prompt_text,
                    prompt_ui,
                    batch_input_ids,
                    batch_prompt_text,
                    batch_prompt_ui,
                ]
            )
            != 1
        ):
            raise ValueError(
                "Must specify exactly one of: input_ids, prompt_text, prompt_ui, batch_input_ids, batch_prompt_text, batch_prompt_ui"
            )

        options = options or self.options

        if prompt_text is not None:
            batch_prompt_text = [prompt_text]
        if prompt_ui is not None:
            batch_prompt_ui = [prompt_ui]
        if input_ids is not None:
            batch_input_ids = [input_ids]
        if batch_prompt_text is not None:
            batch_prompt_ui = [
                [{"type": "text", "value": prompt_text}]
                for prompt_text in batch_prompt_text
            ]
        if batch_prompt_ui is not None:
            batch_input_ids = [
                self.token_manager.tokens_from_ui(prompt_ui)
                for prompt_ui in batch_prompt_ui
            ]

        assert batch_input_ids

        if not options.txt and not options.img:
            raise ValueError("Must specify at least one modality.")
        if options.txt and options.img and len(batch_input_ids) > 1:
            raise ValueError(
                "Batch generation only supported for one modality at a time."
            )

        req_key = self.next_key
        self.next_key += 1

        with self.dctx.active_key_lock:
            self.dctx.active_key[req_key] = True

        self.dctx.req_q.put([options, batch_input_ids, req_key, False])

        try:
            while key_token := self.dctx.res_q.get():
                key, token = key_token
                if key != req_key:
                    # Residual from prior calls to generation. Skip.
                    continue
                if token is None:
                    break
                yield token
        finally:
            with self.dctx.active_key_lock:
                del self.dctx.active_key[req_key]

    def step(self, *args, **kwargs) -> ChameleonGenerator.Token:
        return next(self.stream(*args, **kwargs))

    def generate(self, *args, **kwargs) -> torch.LongTensor:
        tokens = [t.id for t in self.stream(*args, **kwargs)]
        if not tokens:
            return torch.LongTensor()
        return torch.stack(tokens).T

    def decode_text(self, ids: torch.LongTensor | list[list[int]]) -> list[str]:
        return self.token_manager.decode_text(ids)

    def decode_image(self, ids: torch.LongTensor) -> list[PIL.Image]:
        return self.token_manager.decode_image(ids)

    def generate_with_attention(
        self,
        *,
        input_ids: list[int] | None = None,
        prompt_text: str | None = None,
        prompt_ui: list[dict] | None = None,
        batch_input_ids: list[list[int]] | None = None,
        batch_prompt_text: list[str] | None = None,
        batch_prompt_ui: list[list[dict]] | None = None,
        options: Options | None = None,
    ) -> tuple[torch.LongTensor, list[dict]]:
        """
        Generate output tokens and collect attention weights for analysis.
        
        This method uses the same parameters as generate() but also returns
        attention weights for visualization and analysis.
        
        Returns:
            tuple: (output_ids, attention_data)
                - output_ids: tensor of output token IDs
                - attention_data: list of dictionaries with attention weights and metadata
        """
        # First, generate the output tokens
        output_tokens = self.generate(
            input_ids=input_ids,
            prompt_text=prompt_text,
            prompt_ui=prompt_ui,
            batch_input_ids=batch_input_ids,
            batch_prompt_text=batch_prompt_text,
            batch_prompt_ui=batch_prompt_ui,
            options=options,
        )
        
        # Now, determine the input sequence used
        try:
            if prompt_text is not None:
                input_sequence = self.token_manager.tokenize_text(prompt_text)
            elif prompt_ui is not None:
                input_sequence = self.token_manager.tokens_from_ui(prompt_ui)
            elif input_ids is not None:
                input_sequence = input_ids
            elif batch_prompt_text is not None and len(batch_prompt_text) > 0:
                input_sequence = self.token_manager.tokenize_text(batch_prompt_text[0])
            elif batch_prompt_ui is not None and len(batch_prompt_ui) > 0:
                input_sequence = self.token_manager.tokens_from_ui(batch_prompt_ui[0])
            elif batch_input_ids is not None and len(batch_input_ids) > 0:
                input_sequence = batch_input_ids[0]
            else:
                raise ValueError("No valid input provided")
            
            # Only proceed if we have output tokens
            if output_tokens.numel() == 0:
                print("Warning: No output tokens generated")
                return output_tokens, []
                
            # Create the full sequence (input + output)
            output_list = output_tokens.flatten().tolist()
            
            # Don't duplicate the input tokens if they're already in the output
            if len(output_list) >= len(input_sequence) and output_list[:len(input_sequence)] == input_sequence:
                full_sequence = output_list
            else:
                full_sequence = input_sequence + output_list[len(input_sequence):]
            
            # The prompt length is the length of the input sequence
            prompt_length = len(input_sequence)
                
            # Limit sequence length if too long (to avoid memory issues)
            max_analysis_length = 1024  # Set a reasonable limit for analysis
            if len(full_sequence) > max_analysis_length:
                print(f"Warning: Truncating sequence for attention analysis from {len(full_sequence)} to {max_analysis_length}")
                # Preserve the beginning and end portions
                keep_prompt = min(prompt_length, max_analysis_length // 2)
                keep_output = max_analysis_length - keep_prompt
                full_sequence = full_sequence[:keep_prompt] + full_sequence[-keep_output:]
                prompt_length = keep_prompt
            
            # Request attention weights
            attention_data = self._collect_attention_weights(
                full_sequence, 
                prompt_length=prompt_length
            )
            
            return output_tokens, attention_data
            
        except Exception as e:
            print(f"Error collecting attention data: {e}")
            import traceback
            traceback.print_exc()
            return output_tokens, []
    
    def _collect_attention_weights(self, full_sequence: list[int], prompt_length: int) -> list[dict]:
        """
        Collect attention weights for a given sequence.
        
        Args:
            full_sequence: List of token IDs (input + output)
            prompt_length: Length of the input prompt
            
        Returns:
            List of dictionaries with attention data for each layer
        """
        # We need to create a custom message to send to the worker to run
        # a forward pass with output_attention=True
        req_key = self.next_key
        self.next_key += 1
        
        with self.dctx.active_key_lock:
            self.dctx.active_key[req_key] = True
            
        # Custom request for attention collection
        # We'll use a special flag in the options to signal attention collection
        attention_options = Options()
        if isinstance(self.options, Options):
            # Copy options
            for key, value in vars(self.options).items():
                setattr(attention_options, key, value)
        
        # Set a special attribute to signal attention collection
        setattr(attention_options, "collect_attention", True)
        setattr(attention_options, "target_sequence", full_sequence)
        setattr(attention_options, "prompt_length", prompt_length)
        
        try:
            # Send the request
            self.dctx.req_q.put([attention_options, [full_sequence], req_key, False])
            
            # Wait for response which should contain attention data
            attention_data = None
            while True:
                response = self.dctx.res_q.get()
                if response is None:
                    break
                    
                key, data = response
                if key != req_key:
                    # Skip responses from other requests
                    continue
                    
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # This is our attention data
                    attention_data = data
                    break
        finally:
            with self.dctx.active_key_lock:
                if req_key in self.dctx.active_key:
                    del self.dctx.active_key[req_key]
        
        return attention_data or []

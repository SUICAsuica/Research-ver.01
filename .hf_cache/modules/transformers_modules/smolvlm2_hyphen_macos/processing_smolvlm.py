"""Minimal SmolVLM processor implementation for offline MLX inference.

This mirrors enough of the Hugging Face `SmolVLMProcessor` API to let `mlx_vlm`
load the model without requiring PyTorch or the official video processor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, make_nested_list_of_images
from transformers.processing_utils import AllKwargsForChatTemplate, ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding, TextInput
from transformers.utils import logging


logger = logging.get_logger(__name__)


def _prompt_split_image(
    image_seq_len: int,
    image_rows: int,
    image_cols: int,
    fake_token_around_image: str,
    image_token: str,
    global_image_token: str,
) -> str:
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}"
                + f"<row_{n_h + 1}_col_{n_w + 1}>"
                + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_image_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(
    image_seq_len: int,
    fake_token_around_image: str,
    image_token: str,
    global_image_token: str,
) -> str:
    return (
        f"{fake_token_around_image}"
        + f"{global_image_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def _prompt_with_images(
    image_rows: int,
    image_cols: int,
    image_seq_len: int,
    fake_token_around_image: str,
    image_token: str,
    global_image_token: str,
) -> str:
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_image_token=global_image_token,
        )
    return _prompt_split_image(
        image_seq_len,
        image_rows,
        image_cols,
        fake_token_around_image,
        image_token,
        global_image_token,
    )


class SmolVLMProcessor(ProcessorMixin):
    """Lightweight image-only SmolVLM processor."""

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SmolVLMImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor=None,  # kept for API compatibility
        image_seq_len: int = 169,
        chat_template: Optional[str] = None,
        **kwargs: Any,
    ):
        self.fake_image_token = getattr(tokenizer, "fake_image_token", "<fake_token_around_image>")
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.end_of_utterance_token = getattr(tokenizer, "end_of_utterance_token", "<end_of_utterance>")
        self.global_image_token = getattr(tokenizer, "global_image_token", "<global-img>")
        self.image_seq_len = image_seq_len
        self.video_token = getattr(tokenizer, "video_token", "<video>")

        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _expand_text_for_images(
        self,
        prompts: List[str],
        image_rows: Optional[List[List[int]]],
        image_cols: Optional[List[List[int]]],
    ) -> List[str]:
        default_counts = [text.count(self.image_token) for text in prompts]

        if image_rows is None:
            image_rows = [[0] * count for count in default_counts]
        if image_cols is None:
            image_cols = [[0] * count for count in default_counts]

        expanded: List[str] = []
        for text, rows, cols in zip(prompts, image_rows, image_cols):
            prompt_blocks: List[str] = []
            split_text = text.split(self.image_token)
            for idx in range(len(split_text) - 1):
                prompt_blocks.append(split_text[idx])
                prompt_blocks.append(
                    _prompt_with_images(
                        rows[idx],
                        cols[idx],
                        self.image_seq_len,
                        self.fake_image_token,
                        self.image_token,
                        self.global_image_token,
                    )
                )
            prompt_blocks.append(split_text[-1])
            expanded.append("".join(prompt_blocks))
        return expanded

    def __call__(
        self,
        images: Union[ImageInput, Sequence[ImageInput], Sequence[Sequence[ImageInput]]] = None,
        text: Optional[Union[TextInput, Sequence[TextInput]]] = None,
        audio=None,
        videos=None,
        padding: Union[bool, str] = False,
        return_tensors: Optional[str] = None,
        add_special_tokens: Optional[bool] = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        if images is None and text is None:
            raise ValueError("You must specify at least one of `images` or `text`.")

        add_special_tokens = True if add_special_tokens is None else add_special_tokens

        tokenizer_inputs: List[str] = []
        if text is not None:
            if isinstance(text, str):
                tokenizer_inputs = [text]
            else:
                tokenizer_inputs = list(text)

        tokenizer_kwargs: Dict[str, Any] = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "return_attention_mask": True,
        }
        tokenizer_kwargs.update(kwargs.pop("text_kwargs", {}))

        tokenizer_tensor_type: Optional[str] = "np"
        if return_tensors is not None and return_tensors != "mlx":
            tokenizer_tensor_type = return_tensors
        tokenizer_kwargs["return_tensors"] = tokenizer_tensor_type

        image_outputs: Dict[str, Any] = {}
        if images is not None:
            nested = make_nested_list_of_images(images)
            image_processor_kwargs = {
                "return_row_col_info": True,
            }
            image_processor_kwargs.update(kwargs.pop("images_kwargs", {}))
            processed = self.image_processor(nested, **image_processor_kwargs)

            raw_rows = None
            raw_cols = None
            if isinstance(processed, BatchFeature):
                raw_rows = processed.pop("rows", processed.pop("image_rows", None))
                raw_cols = processed.pop("cols", processed.pop("image_cols", None))
                image_outputs = dict(processed.items())
            else:
                processed_dict = dict(processed)
                raw_rows = processed_dict.pop("rows", processed_dict.pop("image_rows", None))
                raw_cols = processed_dict.pop("cols", processed_dict.pop("image_cols", None))
                image_outputs = processed_dict

            if tokenizer_inputs:
                tokenizer_inputs = self._expand_text_for_images(
                    tokenizer_inputs,
                    raw_rows,
                    raw_cols,
                )

        text_outputs: Dict[str, Any] = {}
        if tokenizer_inputs:
            encoded = self.tokenizer(tokenizer_inputs, **tokenizer_kwargs)
            text_outputs = dict(encoded.data)
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if key in text_outputs and isinstance(text_outputs[key], list):
                    text_outputs[key] = np.array(text_outputs[key])

        batch: Dict[str, Any] = {}
        batch.update(image_outputs)
        batch.update(text_outputs)

        if return_tensors == "mlx":
            try:
                import mlx.core as mx  # type: ignore
            except ImportError:
                logger.warning("mlx is not installed; falling back to python lists.")
            else:
                convertible_keys = ("input_ids", "attention_mask", "pixel_values", "pixel_attention_mask")
                for key in convertible_keys:
                    if key in batch and not isinstance(batch[key], (str, list)):
                        batch[key] = mx.array(batch[key])
                    elif key in batch and isinstance(batch[key], list):
                        batch[key] = mx.array(batch[key])

        return BatchEncoding(data=batch, tensor_type=None)

    def apply_chat_template(
        self,
        conversation: List[Dict[str, Union[str, List[Dict[str, str]]]]],
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        return super().apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

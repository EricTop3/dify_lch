"""Functionality for splitting text."""
from __future__ import annotations

from typing import Any, Optional

from core.model_manager import ModelInstance
from core.model_runtime.model_providers.__base.tokenizers.gpt2_tokenzier import GPT2Tokenizer
from core.rag.splitter.text_splitter import (
    TS,
    Collection,
    Literal,
    RecursiveCharacterTextSplitter,
    Set,
    TokenTextSplitter,
    Union,
)


class EnhanceRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """
        This class is used to implement from_gpt2_encoder, to prevent using of tiktoken
    """

    @classmethod
    def from_encoder(
            cls: type[TS],
            embedding_model_instance: Optional[ModelInstance],
            allowed_special: Union[Literal[all], Set[str]] = set(),
            disallowed_special: Union[Literal[all], Collection[str]] = "all",
            **kwargs: Any,
    ):
        def _token_encoder(text: str) -> int:
            if not text:
                return 0

            if embedding_model_instance:
                return embedding_model_instance.get_text_embedding_num_tokens(
                    texts=[text]
                )
            else:
                return GPT2Tokenizer.get_num_tokens(text)

        if issubclass(cls, TokenTextSplitter):
            extra_kwargs = {
                "model_name": embedding_model_instance.model if embedding_model_instance else 'gpt2',
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_token_encoder, **kwargs)


class FixedRecursiveCharacterTextSplitter(EnhanceRecursiveCharacterTextSplitter):
    def __init__(self, fixed_separator: str = "\n\n", separators: Optional[list[str]] = None, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._fixed_separator = fixed_separator
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """Split incoming text and return chunks."""
          # 自定义切分的主要差异是会先按照用户指定的分段标识符进行额外切分，但是要特别注意这部分目前的实现比较粗糙：
          # 1、按照用户指定的分段标识符进行切分时，分段重叠参数是不生效的，Dify 会直接按照指定的分段标识符切分；
          # 2、按照用户指定的分段标识符切分时，不会执行分段的合并，可能会产生大量的长度较小的碎片文本；
          # 3、用户指定的分段标识符为空的情况下，执行的 chunks = list(text) 明显是有问题的，会按照单个字母切分字符串，最终完全不可用；

         # 额外指定分段标识符的情况下，会按照用户指定的分隔标识符先切分
        if self._fixed_separator:
            chunks = text.split(self._fixed_separator)
        else:
            chunks = [text]

        final_chunks = []
        for chunk in chunks:
            # 按照用户执行的分段标识符切分后超过阈值，此时按照标识符列表依次递归切分
            if self._length_function(chunk) > self._chunk_size:
                final_chunks.extend(self.recursive_split_text(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def recursive_split_text(self, text: str) -> list[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = self._separators[-1]
        for _s in self._separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        # Now that we have the separator, split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                other_info = self.recursive_split_text(s)
                final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, separator)
            final_chunks.extend(merged_text)
        return final_chunks

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import json
import os
import queue
from pathlib import Path
from typing import TypeAlias, TypeVar

from transformers import AutoTokenizer, PythonBackend, TokenizersBackend

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config

from .protocol import TokenizerLike

HfTokenizer: TypeAlias = PythonBackend | TokenizersBackend
_T = TypeVar("_T", bound=TokenizerLike)
logger = init_logger(__name__)


class ThreadSafeHFTokenizerMixin:
    """Mixin class for thread-safe HF fast tokenizers."""

    pass


def maybe_make_thread_pool(tokenizer: _T, copies: int = 1):
    """If `tokenizer` is a `TokenizersBackend`, modify the tokenizer
    in-place to make the public interface thread-safe by routing calls
    through a deep-copied tokenizer pool.

    Note that:
    - Only ``TokenizerLike``'s public interface is thread-safe.
      This doesn't include ``_tokenizer`` property nor any mutation
      methods like ``add_special_tokens`` or ``add_tokens``.
    - Adjacent method calls could happen on different deep copies.
    """
    if not isinstance(tokenizer, TokenizersBackend) or isinstance(
        tokenizer, ThreadSafeHFTokenizerMixin
    ):
        return tokenizer

    og_tokenizer = copy.copy(tokenizer)

    import threading
    from functools import lru_cache
    from tokenizers import Tokenizer as _RustTokenizer
    from transformers.tokenization_utils_base import (
        BatchEncoding,
        PreTrainedTokenizerBase,
        TruncationStrategy,
    )
    from transformers.utils import PaddingStrategy

    _ALLOWED_NORMALIZER = {
        "type": "Replace",
        "pattern": {"String": " "},
        "content": "\u2581",
    }
    _ALLOWED_PRE_TOKENIZER = {
        "type": "Split",
        "pattern": {"String": " "},
        "behavior": "MergedWithPrevious",
        "invert": False,
    }
    _ALLOWED_PP_SINGLE = [{"Sequence": {"id": "A", "type_id": 0}}]
    _DISALLOWED_PARALLEL_VALS = ("0", "false", "no", "off")

    def _extract_safe_space_chunk_cfg(
        tok_json_str: str,
    ) -> frozenset[str] | bool:
        try:
            cfg = json.loads(tok_json_str)
        except Exception:
            return False
        if cfg.get("normalizer") != _ALLOWED_NORMALIZER:
            return False
        pre = cfg.get("pre_tokenizer")
        if pre is not None and pre != _ALLOWED_PRE_TOKENIZER:
            return False
        pp = cfg.get("post_processor")
        if pp is not None:
            if (
                not isinstance(pp, dict)
                or pp.get("type") != "TemplateProcessing"
                or pp.get("single") != _ALLOWED_PP_SINGLE
                or pp.get("special_tokens")
            ):
                return False
        for tok_entry in cfg.get("added_tokens") or ():
            content = tok_entry.get("content", "")
            if (
                not content
                or " " in content
                or "\u2581" in content
                or tok_entry.get("single_word")
                or tok_entry.get("lstrip")
                or tok_entry.get("rstrip")
                or tok_entry.get("normalized")
            ):
                return False
        model = cfg.get("model")
        if (
            not isinstance(model, dict)
            or model.get("type") != "BPE"
            or model.get("byte_fallback") is not True
            or model.get("fuse_unk") is not True
            or model.get("ignore_merges")
            or model.get("continuing_subword_prefix") is not None
            or model.get("end_of_word_suffix") is not None
            or model.get("dropout")
        ):
            return False

        def _collect_forbidden_prev(seq) -> None:
            for item in seq:
                seen_non_meta = False
                for idx, ch in enumerate(item):
                    if ch != "\u2581":
                        seen_non_meta = True
                    elif seen_non_meta:
                        forbidden_prev.add(item[idx - 1])

        forbidden_prev: set[str] = set()
        _collect_forbidden_prev(model.get("vocab") or ())
        _collect_forbidden_prev(
            f"{m[0]}{m[1]}" if isinstance(m, (list, tuple)) else m.replace(" ", "", 1)
            for m in (model.get("merges") or ())
        )
        return frozenset(forbidden_prev)

    def _split_safe_space_chunks(
        s: str,
        target_chunks: int,
        forbidden_prev: frozenset[str],
    ) -> list[str]:
        n = len(s)
        stride = max(256, n // target_chunks)
        chunks: list[str] = []
        start = 0
        find = s.find
        while True:
            target = start + stride
            if target >= n - 64:
                break
            pos = find(" ", target)
            while pos != -1 and pos < n - 1:
                prev_ch = s[pos - 1]
                if (
                    prev_ch != " "
                    and prev_ch not in forbidden_prev
                    and s[pos + 1] != " "
                ):
                    break
                pos = find(" ", pos + 1)
            if pos == -1 or pos >= n - 1:
                break
            chunks.append(s[start:pos])
            start = pos
        chunks.append(s[start:])
        return chunks

    def _norm_trunc_key(trunc: dict | None):
        if trunc is None:
            return None
        return (
            trunc.get("max_length"),
            trunc.get("stride"),
            trunc.get("strategy"),
            trunc.get("direction"),
        )

    def _norm_pad_key(pad: dict | None):
        if pad is None:
            return None
        return (
            pad.get("length"),
            pad.get("direction"),
            pad.get("pad_id"),
            pad.get("pad_token"),
            pad.get("pad_type_id"),
            pad.get("pad_to_multiple_of"),
        )

    _MAX_SHARED_RUST_TOKS = 16

    @lru_cache(maxsize=32)
    def _get_cow_worker_cls(base_cls: type) -> type:
        class _CoWTokenizersBackend(base_cls):  # type: ignore[misc,valid-type]
            def _materialize_private_rust_tok(self) -> None:
                if not getattr(self, "_is_private_rust_tok", False):
                    base_tok, lock, json_box, _, cow_mutated = self._cow_state
                    cow_mutated[0] = True
                    with lock:
                        if json_box[0] is None:
                            json_box[0] = base_tok.to_str()
                    self._tokenizer = _RustTokenizer.from_str(json_box[0])
                    self._is_private_rust_tok = True
                    self._cow_key = None

            def _get_or_create_shared_tok(
                self,
                key,
                trunc_target: dict | None,
                pad_target: dict | None,
                desired_split: bool,
            ):
                base_tok, lock, json_box, shared_cache, _ = self._cow_state
                tok = shared_cache.get(key)
                if tok is not None:
                    return tok
                with lock:
                    tok = shared_cache.get(key)
                    if tok is not None:
                        return tok
                    if len(shared_cache) >= _MAX_SHARED_RUST_TOKS:
                        return None
                    if json_box[0] is None:
                        json_box[0] = base_tok.to_str()
                    new_tok = _RustTokenizer.from_str(json_box[0])
                    if trunc_target is None:
                        if new_tok.truncation is not None:
                            new_tok.no_truncation()
                    else:
                        new_tok.enable_truncation(**trunc_target)
                    if pad_target is None:
                        if new_tok.padding is not None:
                            new_tok.no_padding()
                    else:
                        new_tok.enable_padding(**pad_target)
                    if bool(new_tok.encode_special_tokens) != desired_split:
                        new_tok.encode_special_tokens = desired_split
                    shared_cache[key] = new_tok
                    return new_tok

            def set_truncation_and_padding(
                self,
                padding_strategy,
                truncation_strategy,
                max_length,
                stride,
                pad_to_multiple_of,
                padding_side,
            ):
                if not getattr(self, "_is_private_rust_tok", False):
                    if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
                        trunc_target = None
                        trunc_key = None
                    else:
                        trunc_target = {
                            "max_length": max_length,
                            "stride": stride,
                            "strategy": truncation_strategy.value,
                            "direction": self.truncation_side,
                        }
                        trunc_key = (
                            max_length,
                            stride,
                            truncation_strategy.value,
                            self.truncation_side,
                        )

                    if padding_strategy == PaddingStrategy.DO_NOT_PAD:
                        pad_target = None
                        pad_key = None
                    else:
                        length = (
                            max_length
                            if padding_strategy == PaddingStrategy.MAX_LENGTH
                            else None
                        )
                        direction = (
                            padding_side
                            if padding_side is not None
                            else self.padding_side
                        )
                        pad_target = {
                            "length": length,
                            "direction": direction,
                            "pad_id": self.pad_token_id,
                            "pad_token": self.pad_token,
                            "pad_type_id": self.pad_token_type_id,
                            "pad_to_multiple_of": pad_to_multiple_of,
                        }
                        pad_key = (
                            length,
                            direction,
                            self.pad_token_id,
                            self.pad_token,
                            self.pad_token_type_id,
                            pad_to_multiple_of,
                        )

                    desired_split = getattr(self, "_desired_split_special", None)
                    if desired_split is None:
                        desired_split = bool(self._tokenizer.encode_special_tokens)
                    key = (trunc_key, pad_key, desired_split)
                    if getattr(self, "_cow_key", None) == key:
                        return
                    shared_tok = self._get_or_create_shared_tok(
                        key, trunc_target, pad_target, desired_split
                    )
                    if shared_tok is not None:
                        self._tokenizer = shared_tok
                        self._cow_key = key
                        return
                    self._materialize_private_rust_tok()
                return super().set_truncation_and_padding(
                    padding_strategy,
                    truncation_strategy,
                    max_length,
                    stride,
                    pad_to_multiple_of,
                    padding_side,
                )

            def _ensure_chunked_cache(self):
                """Lazily parse safe-chunk config and build shared unpadded Rust clone."""
                cfg_box = getattr(self, "_cow_chunk_cfg_box", None)
                if cfg_box is None:
                    return False, None
                cfg = cfg_box[0]
                if cfg is None:
                    base_tok, lock, json_box, _, _ = self._cow_state
                    with lock:
                        if cfg_box[0] is None:
                            if json_box[0] is None:
                                json_box[0] = base_tok.to_str()
                            cfg_box[0] = _extract_safe_space_chunk_cfg(json_box[0])
                        cfg = cfg_box[0]
                if not cfg:
                    return False, None
                chunk_tok = self._get_or_create_shared_tok(
                    (None, None, False), None, None, False
                )
                return cfg, chunk_tok

            def _try_chunked_encode_plus(self, args, kwargs):
                n_args = len(args)
                text = args[0] if n_args > 0 else kwargs.get("text")
                if not isinstance(text, str) or len(text) < 1536:
                    return None
                if (
                    os.environ.get("TOKENIZERS_PARALLELISM", "").lower()
                    in _DISALLOWED_PARALLEL_VALS
                ):
                    return None
                text_pair = args[1] if n_args > 1 else kwargs.get("text_pair")
                if text_pair is not None:
                    return None
                padding_strategy = (
                    args[3]
                    if n_args > 3
                    else kwargs.get("padding_strategy", PaddingStrategy.DO_NOT_PAD)
                )
                if padding_strategy != PaddingStrategy.DO_NOT_PAD:
                    return None
                truncation_strategy = (
                    args[4]
                    if n_args > 4
                    else kwargs.get(
                        "truncation_strategy", TruncationStrategy.DO_NOT_TRUNCATE
                    )
                )
                max_length = args[5] if n_args > 5 else kwargs.get("max_length")
                stride = args[6] if n_args > 6 else kwargs.get("stride", 0)
                is_split_into_words = (
                    args[7]
                    if n_args > 7
                    else kwargs.get("is_split_into_words", False)
                )
                return_tensors = (
                    args[10] if n_args > 10 else kwargs.get("return_tensors")
                )
                return_token_type_ids = (
                    args[11] if n_args > 11 else kwargs.get("return_token_type_ids")
                )
                return_attention_mask = (
                    args[12] if n_args > 12 else kwargs.get("return_attention_mask")
                )
                return_overflowing_tokens = (
                    args[13]
                    if n_args > 13
                    else kwargs.get("return_overflowing_tokens", False)
                )
                return_special_tokens_mask = (
                    args[14]
                    if n_args > 14
                    else kwargs.get("return_special_tokens_mask", False)
                )
                return_offsets_mapping = (
                    args[15]
                    if n_args > 15
                    else kwargs.get("return_offsets_mapping", False)
                )
                return_length = (
                    args[16] if n_args > 16 else kwargs.get("return_length", False)
                )
                verbose = args[17] if n_args > 17 else kwargs.get("verbose", True)

                if (
                    is_split_into_words
                    or return_tensors is not None
                    or return_overflowing_tokens
                    or return_special_tokens_mask
                    or return_offsets_mapping
                    or return_length
                ):
                    return None
                for k in kwargs:
                    if k not in (
                        "text",
                        "text_pair",
                        "add_special_tokens",
                        "padding_strategy",
                        "truncation_strategy",
                        "max_length",
                        "stride",
                        "is_split_into_words",
                        "pad_to_multiple_of",
                        "padding_side",
                        "return_tensors",
                        "return_token_type_ids",
                        "return_attention_mask",
                        "return_overflowing_tokens",
                        "return_special_tokens_mask",
                        "return_offsets_mapping",
                        "return_length",
                        "verbose",
                        "split_special_tokens",
                    ):
                        return None
                trunc_side = self.truncation_side
                if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                    if (
                        stride != 0
                        or not isinstance(max_length, int)
                        or max_length < 0
                        or trunc_side not in ("left", "right")
                        or truncation_strategy
                        not in (
                            TruncationStrategy.LONGEST_FIRST,
                            TruncationStrategy.ONLY_FIRST,
                        )
                    ):
                        return None

                cfg, chunk_tok = self._ensure_chunked_cache()
                if cfg is False or chunk_tok is None:
                    return None

                target_chunks = min(8, max(4, len(text) // 2048))
                chunks = _split_safe_space_chunks(text, target_chunks, cfg)
                if len(chunks) <= 1:
                    return None

                encodings = chunk_tok.encode_batch(chunks, add_special_tokens=False)
                ids: list[int] = []
                extend = ids.extend
                for enc in encodings:
                    extend(enc.ids)

                if (
                    truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
                    and len(ids) > max_length
                ):
                    if max_length == 0:
                        ids = []
                    elif trunc_side == "left":
                        ids = ids[-max_length:]
                    else:
                        ids = ids[:max_length]
                self._eventual_warn_about_too_long_sequence(ids, max_length, verbose)

                data: dict[str, list[int]] = {"input_ids": ids}
                names = self.model_input_names
                n_ids = len(ids)
                if return_token_type_ids or (
                    return_token_type_ids is None and "token_type_ids" in names
                ):
                    data["token_type_ids"] = [0] * n_ids
                if return_attention_mask or (
                    return_attention_mask is None and "attention_mask" in names
                ):
                    data["attention_mask"] = [1] * n_ids
                return BatchEncoding(data)

            def vllm_encode_ids(
                self,
                text: str,
                add_special_tokens: bool = True,
                max_length: int | None = None,
                trunc_side: str | None = None,
            ) -> list[int] | None:
                if (
                    getattr(self, "_is_private_rust_tok", False)
                    or self.split_special_tokens
                    or not isinstance(text, str)
                ):
                    return None
                if trunc_side is None:
                    trunc_side = self.truncation_side
                if max_length is not None and (
                    max_length <= 0 or trunc_side not in ("left", "right")
                ):
                    return None

                if (
                    len(text) >= 1536
                    and os.environ.get("TOKENIZERS_PARALLELISM", "").lower()
                    not in _DISALLOWED_PARALLEL_VALS
                ):
                    cfg, chunk_tok = self._ensure_chunked_cache()
                    if cfg is not False and chunk_tok is not None:
                        target_chunks = min(8, max(4, len(text) // 2048))
                        chunks = _split_safe_space_chunks(text, target_chunks, cfg)
                        if len(chunks) > 1:
                            encodings = chunk_tok.encode_batch(
                                chunks, add_special_tokens=False
                            )
                            ids: list[int] = []
                            extend = ids.extend
                            for enc in encodings:
                                extend(enc.ids)
                            if max_length is not None and len(ids) > max_length:
                                if trunc_side == "left":
                                    ids = ids[-max_length:]
                                else:
                                    ids = ids[:max_length]
                            return ids

                if max_length is None:
                    trunc_target = None
                    trunc_key = None
                else:
                    trunc_target = {
                        "max_length": max_length,
                        "stride": 0,
                        "strategy": "longest_first",
                        "direction": trunc_side,
                    }
                    trunc_key = (max_length, 0, "longest_first", trunc_side)

                shared_tok = self._get_or_create_shared_tok(
                    (trunc_key, None, False),
                    trunc_target,
                    None,
                    False,
                )
                if shared_tok is None:
                    return None
                return shared_tok.encode(
                    text, add_special_tokens=add_special_tokens
                ).ids

            def _encode_plus(self, *args, **kwargs):
                if getattr(self, "_is_private_rust_tok", False):
                    return super()._encode_plus(*args, **kwargs)
                split_special = kwargs.get("split_special_tokens")
                if split_special is None and len(args) > 18:
                    split_special = args[18]
                if split_special is None:
                    split_special = self.split_special_tokens
                if not split_special:
                    fast_out = self._try_chunked_encode_plus(args, kwargs)
                    if fast_out is not None:
                        return fast_out
                self._desired_split_special = bool(split_special)
                try:
                    return super()._encode_plus(*args, **kwargs)
                finally:
                    self._desired_split_special = None

            def _add_tokens(self, new_tokens, special_tokens=False):
                self._materialize_private_rust_tok()
                return super()._add_tokens(new_tokens, special_tokens=special_tokens)

            def update_post_processor(self):
                self._materialize_private_rust_tok()
                return super().update_post_processor()

        _CoWTokenizersBackend.__name__ = f"_CoW{base_cls.__name__}"
        _CoWTokenizersBackend.__qualname__ = base_cls.__qualname__
        return _CoWTokenizersBackend

    _init_split_special = bool(getattr(og_tokenizer, "split_special_tokens", False))
    _init_trunc_side = getattr(og_tokenizer, "truncation_side", "right")
    base_rust_tok = getattr(og_tokenizer, "_tokenizer", None)
    tokenizer_pool: queue.Queue[TokenizersBackend] = queue.Queue()
    _cow_shared_cache: dict[tuple, _RustTokenizer] | None = None
    _cow_mutated: list[bool] = [False]
    if isinstance(base_rust_tok, _RustTokenizer):
        cow_cls = _get_cow_worker_cls(type(og_tokenizer))
        init_key = (
            _norm_trunc_key(base_rust_tok.truncation),
            _norm_pad_key(base_rust_tok.padding),
            bool(base_rust_tok.encode_special_tokens),
        )
        shared_cache: dict[tuple, _RustTokenizer] = {init_key: base_rust_tok}
        _cow_shared_cache = shared_cache
        chunk_cfg_box: list = [None]
        cow_state = (
            base_rust_tok,
            threading.Lock(),
            [None],
            shared_cache,
            _cow_mutated,
        )

        def _make_worker() -> TokenizersBackend:
            worker = object.__new__(cow_cls)
            worker.__dict__ = og_tokenizer.__dict__.copy()
            if hasattr(worker, "deprecation_warnings"):
                worker.deprecation_warnings = dict(worker.deprecation_warnings)
            if hasattr(worker, "files_loaded"):
                worker.files_loaded = list(worker.files_loaded)
            worker._cow_state = cow_state  # type: ignore[attr-defined]
            worker._cow_chunk_cfg_box = chunk_cfg_box  # type: ignore[attr-defined]
            worker._is_private_rust_tok = False  # type: ignore[attr-defined]
            worker._cow_key = init_key  # type: ignore[attr-defined]
            worker._desired_split_special = None  # type: ignore[attr-defined]
            return worker
    else:

        def _make_worker() -> TokenizersBackend:
            return copy.deepcopy(og_tokenizer)

    for _ in range(copies):
        tokenizer_pool.put(_make_worker())

    _sp_map = getattr(og_tokenizer, "__dict__", {}).get("_special_tokens_map")
    _has_sp_map = isinstance(_sp_map, dict)
    _init_eos_tok = (
        _sp_map.get("eos_token", _SP_CACHE_UNSET) if _has_sp_map else _SP_CACHE_UNSET
    )
    _init_bos_tok = (
        _sp_map.get("bos_token", _SP_CACHE_UNSET) if _has_sp_map else _SP_CACHE_UNSET
    )
    _init_pad_tok = (
        _sp_map.get("pad_token", _SP_CACHE_UNSET) if _has_sp_map else _SP_CACHE_UNSET
    )
    cached_eos_token_id = getattr(og_tokenizer, "eos_token_id", _SP_CACHE_UNSET)
    cached_bos_token_id = getattr(og_tokenizer, "bos_token_id", _SP_CACHE_UNSET)
    cached_pad_token_id = getattr(og_tokenizer, "pad_token_id", _SP_CACHE_UNSET)

    @contextlib.contextmanager
    def _borrow_from_pool():
        try:
            tok = tokenizer_pool.get_nowait()
            yield tok
        except queue.Empty:
            tok = _make_worker()
            yield tok
        finally:
            tokenizer_pool.put(tok)

    def _borrow_attr(attr: str):
        with _borrow_from_pool() as tok:
            return getattr(tok, attr)

    class TokenizerPool(tokenizer.__class__, ThreadSafeHFTokenizerMixin):  # type: ignore
        eos_token_id = _make_cached_sp_id_property(
            "eos_token",
            "_vllm_eos_token_id",
            "_vllm_mut_eos",
            _has_sp_map,
            _init_eos_tok,
            cached_eos_token_id,
            lambda _s: _borrow_attr("eos_token_id"),
        )
        bos_token_id = _make_cached_sp_id_property(
            "bos_token",
            "_vllm_bos_token_id",
            "_vllm_mut_bos",
            _has_sp_map,
            _init_bos_tok,
            cached_bos_token_id,
            lambda _s: _borrow_attr("bos_token_id"),
        )
        pad_token_id = _make_cached_sp_id_property(
            "pad_token",
            "_vllm_pad_token_id",
            "_vllm_mut_pad",
            _has_sp_map,
            _init_pad_tok,
            cached_pad_token_id,
            lambda _s: _borrow_attr("pad_token_id"),
        )

        def apply_chat_template(self, *args, **kwargs):
            try:
                tok = tokenizer_pool.get_nowait()
            except queue.Empty:
                tok = _make_worker()
            try:
                return tok.apply_chat_template(*args, **kwargs)
            finally:
                tokenizer_pool.put(tok)

        def batch_decode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_decode(*args, **kwargs)

        def batch_encode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_encode(*args, **kwargs)

        def convert_tokens_to_ids(self, *args, **kwargs):
            try:
                tok = tokenizer_pool.get_nowait()
            except queue.Empty:
                tok = _make_worker()
            try:
                return tok.convert_tokens_to_ids(*args, **kwargs)
            finally:
                tokenizer_pool.put(tok)

        def convert_ids_to_tokens(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_ids_to_tokens(*args, **kwargs)

        def convert_tokens_to_string(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_tokens_to_string(*args, **kwargs)

        def decode(self, *args, **kwargs):
            try:
                tok = tokenizer_pool.get_nowait()
            except queue.Empty:
                tok = _make_worker()
            try:
                return tok.decode(*args, **kwargs)
            finally:
                tokenizer_pool.put(tok)

        def encode(self, *args, **kwargs):
            try:
                tok = tokenizer_pool.get_nowait()
            except queue.Empty:
                tok = _make_worker()
            try:
                return tok.encode(*args, **kwargs)
            finally:
                tokenizer_pool.put(tok)

        def vllm_encode_ids(
            self,
            text: str,
            add_special_tokens: bool = True,
            max_length: int | None = None,
            trunc_side: str | None = None,
        ) -> list[int] | None:
            if self.__dict__.get("split_special_tokens", _init_split_special):
                return None
            if (
                _cow_shared_cache is not None
                and not _cow_mutated[0]
                and type(text) is str
                and len(text) < 1536
                and self.__dict__.get("_tokenizer") is base_rust_tok
            ):
                if max_length is None:
                    rust_tok = _cow_shared_cache.get((None, None, False))
                    if rust_tok is not None:
                        return rust_tok.encode(
                            text, add_special_tokens=add_special_tokens
                        ).ids
                elif max_length > 0:
                    ts = (
                        trunc_side
                        if trunc_side is not None
                        else self.__dict__.get("truncation_side", _init_trunc_side)
                    )
                    if ts in ("left", "right"):
                        rust_tok = _cow_shared_cache.get(
                            ((max_length, 0, "longest_first", ts), None, False)
                        )
                        if rust_tok is not None:
                            return rust_tok.encode(
                                text, add_special_tokens=add_special_tokens
                            ).ids
                    else:
                        return None
                else:
                    return None
            if max_length is not None and trunc_side is None:
                trunc_side = self.__dict__.get("truncation_side", _init_trunc_side)
            try:
                tok = tokenizer_pool.get_nowait()
            except queue.Empty:
                tok = _make_worker()
            try:
                encode_ids = getattr(tok, "vllm_encode_ids", None)
                if encode_ids is None:
                    return None
                return encode_ids(text, add_special_tokens, max_length, trunc_side)
            finally:
                tokenizer_pool.put(tok)

        def __call__(self, *args, **kwargs):
            try:
                tok = tokenizer_pool.get_nowait()
            except queue.Empty:
                tok = _make_worker()
            try:
                return tok(*args, **kwargs)
            finally:
                tokenizer_pool.put(tok)

        def warmup_chunked_cache(self) -> None:
            with _borrow_from_pool() as tok:
                ensure = getattr(tok, "_ensure_chunked_cache", None)
                if ensure is not None:
                    ensure()

        def _add_tokens(self, new_tokens, special_tokens=False):
            _cow_mutated[0] = True
            return super()._add_tokens(new_tokens, special_tokens=special_tokens)

        def update_post_processor(self):
            _cow_mutated[0] = True
            return super().update_post_processor()

        def __reduce__(self):
            return maybe_make_thread_pool, (og_tokenizer, copies)

    TokenizerPool.__name__ = f"TokenizerPool{og_tokenizer.__class__.__name__}"

    is_std_apply = (
        getattr(type(og_tokenizer), "apply_chat_template", None)
        is PreTrainedTokenizerBase.apply_chat_template
        and "apply_chat_template" not in getattr(og_tokenizer, "__dict__", ())
    )
    tokenizer.__class__ = TokenizerPool
    tokenizer._vllm_std_apply_chat_template = is_std_apply  # type: ignore[attr-defined]
    # Return the tokenizer: TokenizerPool.__reduce__ reconstructs through this
    # function, so falling off the end would unpickle to None (issue #45433).
    return tokenizer


_SP_CACHE_UNSET = object()


def _make_cached_sp_id_property(
    sp_key: str,
    ov_key: str,
    mut_key: str,
    has_sp_map: bool,
    init_tok: object,
    cached_id: object,
    fallback_getter=None,
):
    @property
    def _prop(self):
        if has_sp_map:
            sp_map = self.__dict__.get("_special_tokens_map")
            cur_tok = (
                sp_map.get(sp_key, _SP_CACHE_UNSET)
                if isinstance(sp_map, dict)
                else _SP_CACHE_UNSET
            )
            if cur_tok is init_tok or (
                cur_tok is not _SP_CACHE_UNSET
                and type(cur_tok) is type(init_tok)
                and cur_tok == init_tok
            ):
                if cached_id is not _SP_CACHE_UNSET:
                    return cached_id
                if fallback_getter is not None:
                    return fallback_getter(self)
                return None
            if cur_tok is None:
                return None
            if cur_tok is not _SP_CACHE_UNSET:
                mut = self.__dict__.get(mut_key)
                if mut is not None and (
                    mut[0] is cur_tok
                    or (type(mut[0]) is type(cur_tok) and mut[0] == cur_tok)
                ):
                    return mut[1]
                val = self.convert_tokens_to_ids(str(cur_tok))
                self.__dict__[mut_key] = (cur_tok, val)
                return val
        ov = self.__dict__.get(ov_key, _SP_CACHE_UNSET)
        if ov is not _SP_CACHE_UNSET:
            return ov
        if cached_id is not _SP_CACHE_UNSET:
            return cached_id
        if fallback_getter is not None:
            return fallback_getter(self)
        return None

    @_prop.setter
    def _prop(self, value):
        self.__dict__[ov_key] = value

    return _prop


def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)
    # The underlying tokenizer class could be a specific backend,
    # which does not always implement is_fast in Transformers
    tokenizer_is_fast = getattr(tokenizer, "is_fast", True)

    max_token_id = max(tokenizer_vocab.values())
    max_chars_per_token = max(len(tok) for tok in tokenizer_vocab)

    _sp_map = getattr(tokenizer, "__dict__", {}).get("_special_tokens_map")
    _has_sp_map = isinstance(_sp_map, dict)
    _init_eos_tok = (
        _sp_map.get("eos_token", _SP_CACHE_UNSET) if _has_sp_map else _SP_CACHE_UNSET
    )
    _init_bos_tok = (
        _sp_map.get("bos_token", _SP_CACHE_UNSET) if _has_sp_map else _SP_CACHE_UNSET
    )
    _init_pad_tok = (
        _sp_map.get("pad_token", _SP_CACHE_UNSET) if _has_sp_map else _SP_CACHE_UNSET
    )
    cached_eos_id = getattr(tokenizer, "eos_token_id", None)
    cached_bos_id = getattr(tokenizer, "bos_token_id", None)
    cached_pad_id = getattr(tokenizer, "pad_token_id", None)

    # Some tokenizers (e.g., QwenTokenizer) have special tokens that
    # are added and included in the implementation of the vocab_size
    # property, but not in get_vocab(); if there is an implementation
    # of vocab size, we should take the greater value. vocab_size is a
    # count, so the largest id it implies is vocab_size - 1.
    if hasattr(tokenizer, "vocab_size"):
        with contextlib.suppress(NotImplementedError):
            max_token_id = max(max_token_id, tokenizer.vocab_size - 1)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        eos_token_id = _make_cached_sp_id_property(
            "eos_token",
            "_vllm_cached_eos_id",
            "_vllm_mut_eos",
            _has_sp_map,
            _init_eos_tok,
            cached_eos_id,
        )
        bos_token_id = _make_cached_sp_id_property(
            "bos_token",
            "_vllm_cached_bos_id",
            "_vllm_mut_bos",
            _has_sp_map,
            _init_bos_tok,
            cached_bos_id,
        )
        pad_token_id = _make_cached_sp_id_property(
            "pad_token",
            "_vllm_cached_pad_id",
            "_vllm_mut_pad",
            _has_sp_map,
            _init_pad_tok,
            cached_pad_id,
        )

        @property
        def all_special_ids(self) -> list[int]:
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self) -> list[str]:
            return tokenizer_all_special_tokens

        @property
        def max_token_id(self) -> int:
            return max_token_id

        @property
        def max_chars_per_token(self) -> int:
            return max_chars_per_token

        @property
        def is_fast(self) -> bool:
            return tokenizer_is_fast

        def get_vocab(self) -> dict[str, int]:
            return tokenizer_vocab

        def __len__(self) -> int:
            return tokenizer_len

        def __reduce__(self):
            return get_cached_tokenizer, (tokenizer,)

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    cached_tokenizer.__class__ = CachedTokenizer
    return cached_tokenizer


class CachedHfTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> HfTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path_or_repo_id,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                cache_dir=download_dir,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI. If the "
                    "model was created with a newer version of "
                    "transformers, consider upgrading: "
                    "`uv pip install --upgrade transformers`"
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(
            path_or_repo_id, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
            "do_lower_case", False
        ):
            special_tokens_map = {
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        return get_cached_tokenizer(tokenizer)

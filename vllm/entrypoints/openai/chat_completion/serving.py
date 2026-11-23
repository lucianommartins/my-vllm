# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import itertools
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import Any, Final, cast

import orjson
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    FunctionCall,
    PerRequestMetrics,
    RequestResponseMetadata,
)
from vllm.entrypoints.generate.base.serving import (
    GenerateBaseServing,
    build_per_request_timing_metrics,
    build_spec_decoding_metrics,
    chat_top_logprobs_limit,
    clamp_prompt_logprobs,
    format_token_id_placeholder,
)
from vllm.entrypoints.openai.chat_completion.message_assembly import (
    assemble_chat_message_fields,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.stream_processor import (
    ChatStreamProcessor,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import (
    CompletionTokenUsageInfo,
    ErrorResponse,
    PromptTokenUsageInfo,
    UsageInfo,
)
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens, should_include_usage
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    resolve_finish_reason,
)
from vllm.exceptions import GenerationError
from vllm.inputs import EngineInput, MultiModalPlaceholders
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.renderers.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import as_list
from vllm.utils.serial_utils import numpy2base64

logger = init_logger(__name__)

_SIMPLE_DELTA_FIELDS: Final = frozenset(("role", "content", "reasoning"))
_SSE_DONE_BYTES = b"data: [DONE]\n\n"


def _fast_chat_stream_json(
    *,
    request_id: str,
    chunk_object_type: str,
    created_time: int,
    model_name: str,
    choice_dict: dict[str, Any],
    kv_transfer_params: dict[str, Any] | None = None,
    has_kv_transfer_params: bool = False,
    system_fingerprint: str | None = None,
    prompt_token_ids: list[int] | None = None,
    has_prompt_token_ids: bool = False,
    prompt_text: str | None = None,
    has_prompt_text: bool = False,
    _sse_bytes: bool = False,
    _sse_wrap: bool = False,
) -> str | bytes:
    payload: dict[str, Any] = {
        "id": request_id,
        "object": chunk_object_type,
        "created": created_time,
        "choices": [choice_dict],
        "model": model_name,
    }
    if has_kv_transfer_params:
        payload["kv_transfer_params"] = kv_transfer_params
    if system_fingerprint is not None:
        payload["system_fingerprint"] = system_fingerprint
    if has_prompt_token_ids:
        payload["prompt_token_ids"] = prompt_token_ids
    if has_prompt_text:
        payload["prompt_text"] = prompt_text
    raw = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
    if _sse_bytes:
        return b"data: " + raw + b"\n\n"
    raw_str = raw.decode("utf-8")
    return f"data: {raw_str}\n\n" if _sse_wrap else raw_str


def _build_simple_chat_sse_bytes_tpl(
    *,
    request_id: str,
    chunk_object_type: str,
    created_time: int,
    model_name: str,
    choice_index: int,
    delta_field: str,
) -> tuple[bytes, bytes]:
    p1 = orjson.dumps(
        {
            "id": request_id,
            "object": chunk_object_type,
            "created": created_time,
        },
        option=orjson.OPT_SERIALIZE_NUMPY,
    )[:-1]
    p2 = (
        f',"choices":[{{"index":{choice_index},"delta":{{"{delta_field}":'
    ).encode("utf-8")
    p3 = b'},"logprobs":null,"finish_reason":null,"token_ids":null}],'
    p4 = orjson.dumps(
        {"model": model_name, "kv_transfer_params": None},
        option=orjson.OPT_SERIALIZE_NUMPY,
    )[1:]
    return b"data: " + p1 + p2, p3 + p4 + b"\n\n"


def _format_simple_chat_sse(
    tpl_cache: dict[tuple[int, str], tuple[bytes, bytes, str, str]],
    *,
    request_id: str,
    chunk_object_type: str,
    created_time: int,
    model_name: str,
    choice_index: int,
    delta_field: str,
    value: Any,
    sse_bytes: bool,
    sse_wrap: bool = False,
) -> str | bytes:
    """Format a simple decode delta chunk via cached JSON prefix/suffix."""
    key = (choice_index, delta_field)
    tpl = tpl_cache.get(key)
    if tpl is None:
        prefix_b, suffix_b = _build_simple_chat_sse_bytes_tpl(
            request_id=request_id,
            chunk_object_type=chunk_object_type,
            created_time=created_time,
            model_name=model_name,
            choice_index=choice_index,
            delta_field=delta_field,
        )
        prefix_s = prefix_b[6:].decode("utf-8")
        suffix_s = suffix_b[:-2].decode("utf-8")
        tpl = (prefix_b, suffix_b, prefix_s, suffix_s)
        tpl_cache[key] = tpl
    val_bytes = (
        orjson.dumps(value)
        if type(value) is str
        else orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY)
    )
    if sse_bytes:
        return tpl[0] + val_bytes + tpl[1]
    frame_str = f"{tpl[2]}{val_bytes.decode('utf-8')}{tpl[3]}"
    return f"data: {frame_str}\n\n" if sse_wrap else frame_str


def _format_chat_sse_item(
    chunk: ChatCompletionStreamResponse | ErrorResponse,
    sse_bytes: bool,
) -> str | bytes:
    if isinstance(chunk, ErrorResponse):
        data = chunk.model_dump_json()
    else:
        data = chunk.model_dump_json(
            exclude_unset=True, exclude_none=not chunk.choices
        )
    frame = f"data: {data}\n\n"
    return frame.encode("utf-8") if sse_bytes else frame


def _get_mm_token_counts(engine_input: EngineInput) -> dict[str, int]:
    """Sum per-modality placeholder tokens from ``mm_placeholders``.

    Keyed by modality name; ``PlaceholderRange.length`` is the placeholder's
    prompt token span, so each sum matches the placeholder tokens already
    counted in ``usage.prompt_tokens``.
    """
    mm_placeholders = cast(
        MultiModalPlaceholders | None, engine_input.get("mm_placeholders")
    )
    return {
        modality: sum(p.length for p in ranges)
        for modality, ranges in (mm_placeholders or {}).items()
        if ranges
    }


def _make_prompt_tokens_details(
    enable_prompt_tokens_details: bool,
    num_cached_tokens: int | None,
    num_cache_creation_tokens: int | None,
    mm_token_counts: dict[str, int] | None,
) -> PromptTokenUsageInfo | None:
    """Build ``prompt_tokens_details`` from cached + multimodal token counts."""
    if not enable_prompt_tokens_details:
        return None
    if (
        num_cached_tokens is None
        and num_cache_creation_tokens is None
        and not mm_token_counts
    ):
        return None
    return PromptTokenUsageInfo.model_construct(
        cached_tokens=num_cached_tokens,
        created_cache_tokens=num_cache_creation_tokens,
        multimodal_tokens=mm_token_counts or None,
    )


def _make_completion_tokens_details(
    reasoning_tokens: int,
) -> CompletionTokenUsageInfo:
    return CompletionTokenUsageInfo.model_construct(
        reasoning_tokens=reasoning_tokens
    )


class OpenAIServingChat(GenerateBaseServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        online_renderer: "OnlineRenderer",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        tool_strict_level: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = True,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        enable_per_request_metrics: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.online_renderer = online_renderer
        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs
        self.enable_log_deltas = enable_log_deltas

        self.enable_auto_tools: bool = enable_auto_tools
        self._include_reasoning_tokens_details = bool(reasoning_parser)
        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            tool_strict_level=tool_strict_level,
            model_name=self.model_config.model,
            is_harmony=self.model_config.hf_config.model_type == "gpt_oss",
            tokenizer=self.renderer.tokenizer,
        )
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.enable_per_request_metrics = enable_per_request_metrics
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    def _effective_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
                self.online_renderer.reasoning_effort_map,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    def _make_parser(
        self,
        request: ChatCompletionRequest,
        tokenizer: TokenizerLike,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> Parser:
        assert self.parser_cls is not None
        return self.parser_cls(
            tokenizer,
            request.tools,
            chat_template_kwargs=chat_template_kwargs,
            model_config=self.model_config,
        )

    def _engine_chat_template_kwargs(
        self, chat_template_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Subclass hook to narrow ``chat_template_kwargs`` for the engine.

        The same dict is used twice: to build the API-server-side parser
        instance, and to populate
        ``EngineCoreRequest.reasoning_parser_kwargs`` for the engine-core
        side. The latter crosses ZMQ as msgpack, so a handler that stashes
        request-scoped state only the API-server-side parser needs (values
        msgpack can't encode, or payloads not worth shipping) can drop
        those entries here without affecting the in-process parser.

        Must not mutate the argument -- the caller still needs the full
        dict. The default forwards it unchanged.
        """
        return chat_template_kwargs

    def _engine_reasoning_kwargs(
        self,
        request: ChatCompletionRequest,
        parser: Parser | None,
        chat_template_kwargs: dict[str, Any],
        prompt_token_ids: list[int] | None,
    ) -> dict[str, Any]:
        """`reasoning_ended` and `reasoning_parser_kwargs` for `generate()`."""
        reasoning_parser_kwargs = None
        if parser is not None and parser.reasoning_parser is not None:
            reasoning_parser_kwargs = {
                "chat_template_kwargs": self._engine_chat_template_kwargs(
                    chat_template_kwargs
                ),
            }
        return {
            "reasoning_ended": request.resolve_reasoning_ended(
                parser, prompt_token_ids or []
            ),
            "reasoning_parser_kwargs": reasoning_parser_kwargs,
        }

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
        *,
        _build_conversation: bool = True,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OnlineRenderer, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_inputs) on success,
            or an ErrorResponse on failure.

        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        self._preflight(request.n or 1)

        online_renderer = self.online_renderer
        if (
            not _build_conversation
            and type(online_renderer).render_chat is OnlineRenderer.render_chat
            and "render_chat" not in online_renderer.__dict__
        ):
            return await online_renderer.render_chat(
                request, _build_conversation=False
            )
        return await online_renderer.render_chat(request)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
        *,
        _yield_bytes: bool = False,
        _raw_json: bool = False,
    ) -> (
        AsyncGenerator[str | bytes, None]
        | ChatCompletionResponse
        | ErrorResponse
        | bytes
    ):
        """Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_chat_completion(
                request,
                raw_request,
                _yield_bytes=_yield_bytes,
                _raw_json=_raw_json,
            ),
            request,
            raw_request,
        )

    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
        *,
        _yield_bytes: bool = False,
        _raw_json: bool = False,
    ) -> (
        AsyncGenerator[str | bytes, None]
        | ChatCompletionResponse
        | ErrorResponse
        | bytes
    ):
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        chat_template_kwargs: dict[str, Any] | None = None
        parser: Parser | None = None
        if self.parser_cls is not None:
            chat_template_kwargs = self._effective_chat_template_kwargs(request)
            parser = self._make_parser(request, tokenizer, chat_template_kwargs)
        if (
            not request.echo
            and type(self).render_chat_request
            is OpenAIServingChat.render_chat_request
            and "render_chat_request" not in self.__dict__
        ):
            result = await self.render_chat_request(
                request, _build_conversation=False
            )
        else:
            result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata.model_construct(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        mm_token_counts: dict[str, int] | None = None
        for i, engine_input in enumerate(engine_inputs):
            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids
            mm_token_counts = _get_mm_token_counts(engine_input)

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request._raw_max_tokens(),
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )

            self._log_inputs(
                sub_request_id,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
            session_id = self._get_session_id(request, raw_request)

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_input,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    session_id=session_id,
                    priority=self._get_priority(request, raw_request),
                )
            else:
                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=self._get_priority(request, raw_request),
                    data_parallel_rank=data_parallel_rank,
                    session_id=session_id,
                    **self._engine_reasoning_kwargs(
                        request, parser, chat_template_kwargs, prompt_token_ids
                    ),
                )

            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                chat_template_kwargs=chat_template_kwargs,
                mm_token_counts=mm_token_counts,
                _yield_bytes=_yield_bytes,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser=parser,
            mm_token_counts=mm_token_counts,
            _raw_json=_raw_json,
        )

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    def _create_chat_message(self, *args: Any, **kwargs: Any) -> ChatMessage:
        """Construct the response :class:`ChatMessage` for the non-streaming path.

        Subclasses can swap in a specialized `ChatMessage` subclass
        (e.g. `CohereServingChatV2` returning `CohereChatMessage`)
        without duplicating the field selection in
        ``assemble_chat_message_fields``. The default returns a plain
        `ChatMessage`.
        """
        return (
            ChatMessage.model_construct(**kwargs)
            if not args
            else ChatMessage(*args, **kwargs)
        )

    def _finalize_response_message(
        self,
        message: ChatMessage,
        *,
        parser: Parser | None,
    ) -> ChatMessage:
        """Subclass hook to enrich a fully-constructed :class:`ChatMessage`.

        Default is a no-op. Subclasses that need to surface parser-side
        extras (e.g. :class:`CohereServingChatV2` reading grounding
        citations off the reasoning parser and populating
        :class:`CohereChatMessage.citations`) override this to inspect
        ``parser`` and mutate/replace ``message``.
        """
        return message

    def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        chat_template_kwargs: dict[str, Any] | None = None,
        mm_token_counts: dict[str, int] | None = None,
        *,
        _yield_bytes: bool = False,
    ) -> AsyncGenerator[str | bytes, None]:
        return self._iter_chat_stream(  # type: ignore[return-value]
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            chat_template_kwargs=chat_template_kwargs,
            mm_token_counts=mm_token_counts,
            serialize_sse=True,
            _sse_bytes=_yield_bytes,
            _wrap_sse=True,
        )

    async def chat_completion_stream_generator_raw(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        chat_template_kwargs: dict[str, Any] | None = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> AsyncGenerator[ChatCompletionStreamResponse | ErrorResponse, None]:
        """Yield chat chunks or errors without serializing them into SSE frames."""
        async for chunk in self._iter_chat_stream(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            chat_template_kwargs=chat_template_kwargs,
            mm_token_counts=mm_token_counts,
            serialize_sse=False,
        ):
            assert not isinstance(chunk, str)
            yield chunk

    async def _iter_chat_stream(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        chat_template_kwargs: dict[str, Any] | None = None,
        mm_token_counts: dict[str, int] | None = None,
        *,
        serialize_sse: bool = False,
        _sse_bytes: bool = False,
        _wrap_sse: bool = False,
    ) -> AsyncGenerator[ChatCompletionStreamResponse | ErrorResponse | str | bytes, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True
        sse_tpls: dict[tuple[int, str], tuple[bytes, bytes, str, str]] = {}
        wrap_str = _wrap_sse and not _sse_bytes

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        fast_simple_sse_0 = (
            serialize_sse
            and _sse_bytes
            and self.parser_cls is None
            and num_choices == 1
            and not request.return_token_ids
            and not request.return_weight_versions
            and not (self.enable_log_outputs and self.request_logger)
        )
        simple_prefix_0_b: bytes | None = None
        simple_suffix_0_b: bytes | None = None
        orjson_dumps = orjson.dumps
        # TODO: Remove once all reasoning parsers use the Parser Engine.
        generated_token_ids: list[list[int]] = [[] for _ in range(num_choices)]
        previous_reasoning_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        num_cache_creation_tokens = None

        previous_texts = [""] * num_choices

        try:
            if self.parser_cls is not None:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )
                processors = [
                    ChatStreamProcessor(
                        request,
                        self._make_parser(request, tokenizer, chat_template_kwargs),
                        enable_auto_tools=self.enable_auto_tools,
                        parser_cls=self.parser_cls,
                    )
                    for _ in range(num_choices)
                ]
            else:
                processors = [
                    ChatStreamProcessor(
                        request,
                        None,
                        enable_auto_tools=self.enable_auto_tools,
                        parser_cls=None,
                    )
                    for _ in range(num_choices)
                ]
        except Exception as e:
            logger.exception("Error in parser creation.")
            err = self.create_error_response(e)
            if _wrap_sse:
                yield _format_chat_sse_item(err, _sse_bytes)
                yield _SSE_DONE_BYTES if _sse_bytes else "data: [DONE]\n\n"
            else:
                yield err
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        last_res: RequestOutput | None = None
        fast_mid_sse_0 = (
            fast_simple_sse_0
            and not include_continuous_usage
            and not request.logprobs
        )
        fast_mid_toks_0 = 0
        try:
            async for res in result_generator:
                last_res = res
                if (
                    fast_mid_sse_0
                    and not first_iteration
                    and simple_prefix_0_b is not None
                    and not finish_reason_sent[0]
                ):
                    res_outputs = res.outputs
                    if len(res_outputs) == 1:
                        out = res_outputs[0]
                        if (
                            out.finish_reason is None
                            and out.stop_reason is None
                            and out.index == 0
                        ):
                            d_text = out.text
                            d_toks = out.token_ids
                            fast_mid_toks_0 += len(d_toks)
                            if (
                                d_text
                                or d_toks
                                or fast_mid_toks_0
                                or processors[0].num_output_tokens
                            ):
                                val_b = (
                                    orjson_dumps(d_text)
                                    if type(d_text) is str
                                    else orjson_dumps(
                                        d_text, option=orjson.OPT_SERIALIZE_NUMPY
                                    )
                                )
                                yield (
                                    simple_prefix_0_b  # type: ignore[operator]
                                    + val_b
                                    + simple_suffix_0_b  # type: ignore[operator]
                                )
                            continue
                    if fast_mid_toks_0:
                        processors[0].num_output_tokens += fast_mid_toks_0
                        fast_mid_toks_0 = 0
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    num_cache_creation_tokens = res.num_cache_creation_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # ``res.prompt`` is the rendered chat-templated prompt
                    prompt_text = res.prompt if request.return_prompt_text else None

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        if serialize_sse and not include_continuous_usage:
                            yield _fast_chat_stream_json(
                                request_id=request_id,
                                chunk_object_type=chunk_object_type,
                                created_time=created_time,
                                model_name=model_name,
                                choice_dict={
                                    "index": i,
                                    "delta": {"role": role, "content": ""},
                                    "logprobs": None,
                                    "finish_reason": None,
                                },
                                prompt_token_ids=(
                                    res.prompt_token_ids
                                    if request.return_token_ids
                                    else None
                                ),
                                has_prompt_token_ids=True,
                                prompt_text=prompt_text,
                                has_prompt_text=True,
                                _sse_bytes=_sse_bytes,
                                _sse_wrap=wrap_str,
                            )
                            continue
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                            prompt_text=prompt_text,
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo.model_construct(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                                completion_tokens_details=(
                                    _make_completion_tokens_details(0)
                                    if self._include_reasoning_tokens_details
                                    else None
                                ),
                                # Known from the first output, so the first
                                # chunk's prompt split matches the final one.
                                prompt_tokens_details=_make_prompt_tokens_details(
                                    self.enable_prompt_tokens_details,
                                    num_cached_tokens,
                                    num_cache_creation_tokens,
                                    mm_token_counts,
                                ),
                            )

                        yield (
                            _format_chat_sse_item(chunk, _sse_bytes)
                            if _wrap_sse
                            else chunk
                        )

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo.model_construct(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                        completion_tokens_details=(
                                            _make_completion_tokens_details(0)
                                            if self._include_reasoning_tokens_details
                                            else None
                                        ),
                                    )

                                yield (
                                    _format_chat_sse_item(chunk, _sse_bytes)
                                    if _wrap_sse
                                    else chunk
                                )
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    processor = processors[i]
                    parser = processor.parser
                    if finish_reason_sent[i]:
                        continue

                    self._raise_if_error(output.finish_reason, request_id)

                    if request.logprobs:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            logprob_token_ids=request.logprob_token_ids,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not processor.num_output_tokens
                        and output.finish_reason is None
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    include_token_ids = (
                        request.return_token_ids and not processor.hide_metadata
                    )
                    if (
                        serialize_sse
                        and parser is None
                        and logprobs is None
                        and not include_continuous_usage
                        and not request.return_weight_versions
                    ):
                        processor.num_output_tokens += len(output.token_ids)
                        if output.finish_reason is not None:
                            finish_reason = resolve_finish_reason(
                                output.finish_reason,
                                request,
                                processor.tools_streamed,
                            )
                            finish_reason_sent[i] = True
                        else:
                            finish_reason = None

                        if (
                            self.enable_log_outputs
                            and self.request_logger
                        ):
                            previous_texts[i] += delta_text
                            if delta_text and self.enable_log_deltas:
                                self.request_logger.log_outputs(
                                    request_id=request_id,
                                    outputs=delta_text,
                                    output_token_ids=as_list(output.token_ids),
                                    finish_reason=output.finish_reason,
                                    is_streaming=True,
                                    delta=True,
                                )

                        if (
                            output.finish_reason is None
                            and not include_token_ids
                            and not finish_reason_sent[i]
                        ):
                            if fast_simple_sse_0:
                                if simple_prefix_0_b is None:
                                    simple_prefix_0_b, simple_suffix_0_b = (
                                        _build_simple_chat_sse_bytes_tpl(
                                            request_id=request_id,
                                            chunk_object_type=chunk_object_type,
                                            created_time=created_time,
                                            model_name=model_name,
                                            choice_index=0,
                                            delta_field="content",
                                        )
                                    )
                                val_b = (
                                    orjson_dumps(delta_text)
                                    if type(delta_text) is str
                                    else orjson_dumps(
                                        delta_text, option=orjson.OPT_SERIALIZE_NUMPY
                                    )
                                )
                                yield simple_prefix_0_b + val_b + simple_suffix_0_b
                                continue
                            yield _format_simple_chat_sse(
                                sse_tpls,
                                request_id=request_id,
                                chunk_object_type=chunk_object_type,
                                created_time=created_time,
                                model_name=model_name,
                                choice_index=i,
                                delta_field="content",
                                value=delta_text,
                                sse_bytes=_sse_bytes,
                                sse_wrap=wrap_str,
                            )
                            continue
                        if (
                            not delta_text
                            and not output.token_ids
                            and not processor.num_output_tokens
                        ):
                            d_dict: dict[str, Any] = {}
                        else:
                            d_dict = {"content": delta_text}
                        tok_ids = (
                            as_list(output.token_ids) if include_token_ids else None
                        )
                        if output.finish_reason is None:
                            choice_dict: dict[str, Any] = {
                                "index": i,
                                "delta": d_dict,
                                "logprobs": None,
                                "finish_reason": None,
                                "token_ids": tok_ids,
                            }
                        else:
                            choice_dict = {
                                "index": i,
                                "delta": d_dict,
                                "logprobs": None,
                                "finish_reason": finish_reason,
                                "stop_reason": output.stop_reason,
                                "token_ids": tok_ids,
                                "weight_versions": None,
                            }
                        stamp_fp = (
                            self.system_fingerprint
                            if (
                                not include_usage
                                and self.system_fingerprint is not None
                                and finish_reason is not None
                            )
                            else None
                        )
                        yield _fast_chat_stream_json(
                            request_id=request_id,
                            chunk_object_type=chunk_object_type,
                            created_time=created_time,
                            model_name=model_name,
                            choice_dict=choice_dict,
                            kv_transfer_params=(
                                res.kv_transfer_params if finish_reason_sent[i] else None
                            ),
                            has_kv_transfer_params=True,
                            system_fingerprint=stamp_fp,
                            _sse_bytes=_sse_bytes,
                            _sse_wrap=wrap_str,
                        )
                        continue

                    delta_message, finish_reason = processor.step(
                        delta_text,
                        as_list(output.token_ids),
                        output.finish_reason,
                        res.prompt_token_ids,
                    )

                    if self.enable_log_outputs and self.request_logger:
                        previous_texts[i] += delta_text

                    if parser is not None and self._include_reasoning_tokens_details:
                        generated_token_ids[i].extend(output.token_ids)
                        if include_continuous_usage:
                            previous_reasoning_tokens[i] = (
                                parser.count_reasoning_tokens(
                                    tuple(generated_token_ids[i])
                                )
                            )

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    hide_stream_metadata = processor.hide_metadata
                    if hide_stream_metadata:
                        logprobs = None

                    if delta_message is None:
                        # NOTE: If return_token_ids or logprobs are enabled, we
                        # still need to send a chunk even if delta_message is None
                        # to ensure all tokens are included in the response
                        if (
                            output.finish_reason is None
                            and logprobs is None
                            and (not request.return_token_ids or hide_stream_metadata)
                        ):
                            continue
                        delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content_parts = []
                        if delta_message.content:
                            delta_content_parts.append(delta_message.content)
                        if delta_message.reasoning:
                            reasoning = delta_message.reasoning
                            delta_content_parts.append(f"[reasoning: {reasoning}]")
                        if delta_message.tool_calls:
                            tool_args = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments
                            )
                            if tool_args:
                                delta_content_parts.append(f"[tool_calls: {tool_args}]")

                        if delta_content_parts and self.enable_log_deltas:
                            delta_content = " ".join(delta_content_parts)
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    include_token_ids = (
                        request.return_token_ids and not hide_stream_metadata
                    )
                    d_set = delta_message.__pydantic_fields_set__
                    if (
                        serialize_sse
                        and logprobs is None
                        and not include_continuous_usage
                        and not request.return_weight_versions
                        and not delta_message.model_extra
                        and not delta_message.tool_calls
                        and not (d_set - _SIMPLE_DELTA_FIELDS)
                    ):
                        if (
                            output.finish_reason is None
                            and not include_token_ids
                            and not finish_reason_sent[i]
                            and len(d_set) == 1
                        ):
                            delta_field = next(iter(d_set))
                            yield _format_simple_chat_sse(
                                sse_tpls,
                                request_id=request_id,
                                chunk_object_type=chunk_object_type,
                                created_time=created_time,
                                model_name=model_name,
                                choice_index=i,
                                delta_field=delta_field,
                                value=getattr(delta_message, delta_field),
                                sse_bytes=_sse_bytes,
                                sse_wrap=wrap_str,
                            )
                            continue
                        d_dict = {
                            k: getattr(delta_message, k)
                            for k in _SIMPLE_DELTA_FIELDS
                            if k in d_set
                        }
                        tok_ids = (
                            as_list(output.token_ids) if include_token_ids else None
                        )
                        if output.finish_reason is None:
                            choice_dict: dict[str, Any] = {
                                "index": i,
                                "delta": d_dict,
                                "logprobs": None,
                                "finish_reason": None,
                                "token_ids": tok_ids,
                            }
                        else:
                            finish_reason_sent[i] = True
                            choice_dict = {
                                "index": i,
                                "delta": d_dict,
                                "logprobs": None,
                                "finish_reason": finish_reason,
                                "stop_reason": output.stop_reason,
                                "token_ids": tok_ids,
                                "weight_versions": None,
                            }
                        stamp_fp = (
                            self.system_fingerprint
                            if (
                                not include_usage
                                and self.system_fingerprint is not None
                                and choice_dict["finish_reason"] is not None
                            )
                            else None
                        )
                        yield _fast_chat_stream_json(
                            request_id=request_id,
                            chunk_object_type=chunk_object_type,
                            created_time=created_time,
                            model_name=model_name,
                            choice_dict=choice_dict,
                            kv_transfer_params=(
                                res.kv_transfer_params if finish_reason_sent[i] else None
                            ),
                            has_kv_transfer_params=True,
                            system_fingerprint=stamp_fp,
                            _sse_bytes=_sse_bytes,
                            _sse_wrap=wrap_str,
                        )
                        continue

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids) if include_token_ids else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # Send the finish response for each request.n only once
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids) if include_token_ids else None
                            ),
                            weight_versions=(
                                output.weight_versions
                                if request.return_weight_versions
                                else None
                            ),
                        )

                        finish_reason_sent[i] = True

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                        kv_transfer_params=(
                            res.kv_transfer_params if finish_reason_sent[i] else None
                        ),
                    )
                    # Stamp the fingerprint on terminal chunks only (those with
                    # finish_reason set). When ``include_usage`` is on, the
                    # trailing usage chunk below overrides this as the true
                    # final message.
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        chunk.system_fingerprint = self.system_fingerprint

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = processor.num_output_tokens
                        chunk.usage = UsageInfo.model_construct(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                            completion_tokens_details=(
                                _make_completion_tokens_details(
                                    previous_reasoning_tokens[i]
                                )
                                if self._include_reasoning_tokens_details
                                else None
                            ),
                        )

                    yield (
                        _format_chat_sse_item(chunk, _sse_bytes)
                        if _wrap_sse
                        else chunk
                    )

            if fast_mid_toks_0:
                processors[0].num_output_tokens += fast_mid_toks_0
                fast_mid_toks_0 = 0

            if self._include_reasoning_tokens_details and not include_continuous_usage:
                for i, processor in enumerate(processors):
                    parser = processor.parser
                    if parser is not None and generated_token_ids[i]:
                        previous_reasoning_tokens[i] = parser.count_reasoning_tokens(
                            tuple(generated_token_ids[i])
                        )

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(p.num_output_tokens for p in processors)
                final_usage = UsageInfo.model_construct(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                    completion_tokens_details=_make_completion_tokens_details(
                        sum(previous_reasoning_tokens)
                    )
                    if self._include_reasoning_tokens_details
                    else None,
                )
                final_usage.prompt_tokens_details = _make_prompt_tokens_details(
                    self.enable_prompt_tokens_details,
                    num_cached_tokens,
                    num_cache_creation_tokens,
                    mm_token_counts,
                )

                # In streaming, metrics ride on this final usage chunk, which is
                # only emitted when usage reporting is enabled (i.e.
                # ``stream_options.include_usage=true`` or
                # ``--enable-force-include-usage``).
                stream_per_request_metrics: PerRequestMetrics | None = None
                # See note in chat_completion_full_generator: suppress for n>1.
                if (request.n or 1) == 1:
                    if self.enable_per_request_metrics:
                        last_metrics = (
                            last_res.metrics if last_res is not None else None
                        )
                        stream_per_request_metrics = build_per_request_timing_metrics(
                            last_metrics, completion_tokens
                        )
                    spec_stats = build_spec_decoding_metrics(last_res)
                    if spec_stats is not None:
                        if stream_per_request_metrics is None:
                            stream_per_request_metrics = PerRequestMetrics()
                        stream_per_request_metrics.speculative_decoding = spec_stats

                final_usage_chunk = ChatCompletionStreamResponse.model_construct(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    system_fingerprint=self.system_fingerprint,
                    metrics=stream_per_request_metrics,
                )
                yield (
                    _format_chat_sse_item(final_usage_chunk, _sse_bytes)
                    if _wrap_sse
                    else final_usage_chunk
                )

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(p.num_output_tokens for p in processors)
            request_metadata.final_usage_info = UsageInfo.model_construct(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
                completion_tokens_details=_make_completion_tokens_details(
                    sum(previous_reasoning_tokens)
                )
                if self._include_reasoning_tokens_details
                else None,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i, processor in enumerate(processors):
                    num_tokens = processor.num_output_tokens
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {num_tokens} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except GenerationError as e:
            err = self.create_error_response(
                str(e), err_type="InternalServerError", status_code=e.status_code
            )
            yield _format_chat_sse_item(err, _sse_bytes) if _wrap_sse else err
        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            err = self.create_error_response(e)
            yield _format_chat_sse_item(err, _sse_bytes) if _wrap_sse else err
        if _wrap_sse:
            yield _SSE_DONE_BYTES if _sse_bytes else "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        parser: Parser | None = None,
        mm_token_counts: dict[str, int] | None = None,
        *,
        _raw_json: bool = False,
    ) -> ErrorResponse | ChatCompletionResponse | bytes:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        if final_res is None:
            return self.create_error_response(
                "No output received from the engine.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        choices: list[ChatCompletionResponseChoice] = []
        total_reasoning_tokens = 0
        can_raw = (
            _raw_json
            and len(final_res.outputs) == 1
            and not request.logprobs
            and not request.return_weight_versions
            and not (self.enable_log_outputs and self.request_logger)
            and not self.enable_per_request_metrics
            and not self.enable_prompt_tokens_details
            and final_res.prompt_logprobs is None
            and final_res.outputs[0].spec_decode_metrics is None
        )

        role = self.get_chat_request_role(request)
        for i, output in enumerate(final_res.outputs):
            # check for error finish reason and raise GenerationError
            # finish_reason='error' indicates a retryable request-level internal error
            self._raise_if_error(output.finish_reason, request_id)
            if i > 0 and parser is not None:
                # Parsers are stateful: like streaming, use one per choice.
                parser = self._make_parser(
                    request, tokenizer, self._effective_chat_template_kwargs(request)
                )
            if parser is not None and final_res.prompt_token_ids is not None:
                parser.set_prompt_token_ids(final_res.prompt_token_ids)
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    logprob_token_ids=request.logprob_token_ids,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if parser is not None:
                reasoning, content, tool_calls = parser.parse(
                    output.text,
                    request,
                    enable_auto_tools=self.enable_auto_tools,
                    model_output_token_ids=token_ids,
                )
                suppress_metadata = not request.include_reasoning and parser is not None
                total_reasoning_tokens += parser.count_reasoning_tokens(token_ids)
                if suppress_metadata:
                    logprobs = None
            else:
                reasoning = None
                content = output.text
                tool_calls = []
                suppress_metadata = False

            fields = assemble_chat_message_fields(
                request,
                reasoning,
                content,
                tool_calls,
                enable_auto_tools=self.enable_auto_tools,
                parser_cls=self.parser_cls,
            )
            routed_experts_b64 = (
                numpy2base64(output.routed_experts)
                if output.routed_experts is not None
                else None
            )
            if can_raw and not fields.tool_calls:
                msg_content = fields.content
                if request.echo:
                    last_msg_content: str | list[dict[str, str]] = ""
                    if (
                        conversation
                        and "content" in conversation[-1]
                        and conversation[-1].get("role") == role
                    ):
                        last_msg_content = conversation[-1]["content"] or ""
                    if isinstance(last_msg_content, list):
                        last_msg_content = "\n".join(
                            msg["text"] for msg in last_msg_content
                        )
                    msg_content = last_msg_content + (msg_content or "")
                assert final_res.prompt_token_ids is not None
                num_prompt_tokens = len(final_res.prompt_token_ids)
                if final_res.encoder_prompt_token_ids is not None:
                    num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
                num_generated_tokens = len(output.token_ids)
                payload = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created_time,
                    "model": model_name,
                    "choices": [
                        {
                            "index": output.index,
                            "message": {
                                "role": role,
                                "content": msg_content,
                                "refusal": None,
                                "audio": None,
                                "function_call": None,
                                "reasoning": fields.reasoning,
                            },
                            "logprobs": None,
                            "finish_reason": resolve_finish_reason(
                                output.finish_reason, request, False
                            ),
                            "stop_reason": output.stop_reason,
                            "token_ids": (
                                as_list(output.token_ids)
                                if request.return_token_ids and not suppress_metadata
                                else None
                            ),
                            "routed_experts": routed_experts_b64,
                            "weight_versions": None,
                        }
                    ],
                    "service_tier": None,
                    "system_fingerprint": self.system_fingerprint,
                    "usage": {
                        "prompt_tokens": num_prompt_tokens,
                        "total_tokens": num_prompt_tokens + num_generated_tokens,
                        "completion_tokens": num_generated_tokens,
                        "completion_tokens_details": (
                            {"reasoning_tokens": total_reasoning_tokens}
                            if self._include_reasoning_tokens_details
                            else None
                        ),
                    },
                    "prompt_logprobs": None,
                    "prompt_token_ids": (
                        final_res.prompt_token_ids
                        if request.return_token_ids
                        else None
                    ),
                    "prompt_text": (
                        final_res.prompt if request.return_prompt_text else None
                    ),
                    "kv_transfer_params": final_res.kv_transfer_params,
                    "ec_transfer_params": final_res.ec_transfer_params,
                    "metrics": None,
                }
                return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

            message = self._create_chat_message(
                role=role,
                reasoning=fields.reasoning,
                content=fields.content,
                tool_calls=fields.tool_calls,
            )

            # Subclass hook: enrich the constructed message with any
            # parser-side extras that don't fit through the plain
            # ``(reasoning, content, tool_calls)`` tuple. Base is a no-op;
            # citation-aware handlers use this to surface grounding
            # metadata cached on the reasoning parser.
            message = self._finalize_response_message(message, parser=parser)

            choice_data = ChatCompletionResponseChoice.model_construct(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason=resolve_finish_reason(
                    output.finish_reason, request, bool(message.tool_calls)
                ),
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids)
                    if request.return_token_ids and not suppress_metadata
                    else None
                ),
                routed_experts=routed_experts_b64,
                weight_versions=(
                    output.weight_versions if request.return_weight_versions else None
                ),
            )

            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo.model_construct(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            completion_tokens_details=_make_completion_tokens_details(
                total_reasoning_tokens
            )
            if self._include_reasoning_tokens_details
            else None,
        )
        usage.prompt_tokens_details = _make_prompt_tokens_details(
            self.enable_prompt_tokens_details,
            final_res.num_cached_tokens,
            final_res.num_cache_creation_tokens,
            mm_token_counts,
        )

        request_metadata.final_usage_info = usage

        per_request_metrics: PerRequestMetrics | None = None
        # Per-request metrics (timing + spec-decode acceptance) describe a single
        # generation stream. For n>1 the stats belong to only one of the n
        # sequences, so they cannot be attributed to the request; suppress.
        if (request.n or 1) == 1:
            if self.enable_per_request_metrics:
                per_request_metrics = build_per_request_timing_metrics(
                    final_res.metrics, num_generated_tokens
                )
            spec_stats = build_spec_decoding_metrics(final_res)
            if spec_stats is not None:
                if per_request_metrics is None:
                    per_request_metrics = PerRequestMetrics()
                per_request_metrics.speculative_decoding = spec_stats

        # ``final_res.prompt`` is the rendered chat-templated prompt text
        prompt_text = final_res.prompt if request.return_prompt_text else None

        response = ChatCompletionResponse.model_construct(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=self.system_fingerprint,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            prompt_text=prompt_text,
            kv_transfer_params=final_res.kv_transfer_params,
            ec_transfer_params=final_res.ec_transfer_params,
            metrics=per_request_metrics,
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:  # type: ignore
                        function_call: FunctionCall = tc.function  # type: ignore
                        tool_call_descriptions.append(
                            f"{function_call.name}({function_call.arguments})"
                        )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: TokenizerLike | None,
        should_return_as_token_id: bool,
        return_all: bool = False,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for p in itertools.islice(
                logprobs.items(), chat_top_logprobs_limit(top_logprobs, return_all)
            )
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: TokenizerLike | None,
        num_output_top_logprobs: int | None = None,
        logprob_token_ids: list[int] | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = format_token_id_placeholder(token_id)
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Unable to get tokenizer because `skip_tokenizer_init=True`"
                        )

                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                            return_all=bool(logprob_token_ids),
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)

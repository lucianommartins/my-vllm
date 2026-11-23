# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import json
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_serializer,
)
from pydantic_core import PydanticUndefined

import vllm.envs as envs
from vllm.config.utils import replace
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.sampling_params import StructuredOutputsParams

logger = init_logger(__name__)

StopParam: TypeAlias = (
    str | Annotated[list[str], Field(max_length=envs.VLLM_MAX_STOP_STRINGS)] | None
)

# `top_logprobs` is nullable in the OpenAI spec; null means the same as omitted.
TopLogprobsParam: TypeAlias = Annotated[
    int,
    BeforeValidator(lambda v: 0 if v is None else v, json_schema_input_type=int | None),
]

MMPayload: TypeAlias = Literal["tensors", "none"]

_CACHE_SALT_FORBIDDEN_CHARS = frozenset("@/\\\x00")
_MAX_CACHE_SALT_LENGTH = 128


def validate_cache_salt(cache_salt: object) -> None:
    """Validate cache salts before they reach downstream cache backends."""
    if cache_salt is None:
        return
    if not isinstance(cache_salt, str) or not cache_salt:
        raise VLLMValidationError(
            "Parameter 'cache_salt' must be a non-empty string if provided.",
            parameter="cache_salt",
        )
    if len(cache_salt) > _MAX_CACHE_SALT_LENGTH or any(
        char in _CACHE_SALT_FORBIDDEN_CHARS for char in cache_salt
    ):
        raise VLLMValidationError(
            "Parameter 'cache_salt' must be at most 128 characters and must "
            "not contain '@', '/', '\\\\', or NUL.",
            parameter="cache_salt",
        )


def resolve_return_mm_kwargs_alias(data: Any) -> Any:
    """Map the deprecated `return_mm_kwargs=False` onto `mm_payload="none"`."""
    if (
        isinstance(data, dict)
        and data.get("return_mm_kwargs") is False
        and "mm_payload" not in data
    ):
        data = {**data, "mm_payload": "none"}
    return data


def validate_request_mm_kwargs(
    *,
    mm_processor_kwargs: dict[str, Any] | None,
    media_io_kwargs: dict[str, dict[str, Any]] | None,
    trust_request_mm_kwargs: bool,
) -> None:
    """Reject untrusted per-request multimodal kwarg overrides."""
    if trust_request_mm_kwargs:
        return

    if mm_processor_kwargs:
        raise VLLMValidationError(
            "Per-request mm_processor_kwargs are disabled by default because "
            "they can change multimodal preprocessing resource usage. Start "
            "the server with --trust-request-mm-kwargs only when clients "
            "are trusted.",
            parameter="mm_processor_kwargs",
        )
    if media_io_kwargs:
        raise VLLMValidationError(
            "Per-request media_io_kwargs are disabled by default because "
            "they can change multimodal media loading resource usage. Start "
            "the server with --trust-request-mm-kwargs only when clients "
            "are trusted.",
            parameter="media_io_kwargs",
        )


class OpenAIResponseModel(BaseModel):
    """Lightweight BaseModel for internal/outbound response models (no extra-field logging)."""

    model_config = ConfigDict(extra="allow")
    field_names: ClassVar[set[str] | None] = None


class SpeculativeDecodingMetrics(OpenAIResponseModel):
    """Per-request speculative-decoding acceptance metrics.

    Experimental, subject to change. Only populated for single-sequence requests
    (`n == 1`); `null` for `n > 1`, mirroring the timing metrics.
    """

    mean_acceptance_length: float
    draft_acceptance_rate: float
    # Dense histogram: index j holds the number of verify steps that accepted
    # exactly j draft tokens (length num_spec_tokens + 1). Excludes the
    # always-accepted bonus token.
    acceptance_histogram: list[int]
    num_spec_steps: int
    num_accepted_draft_tokens: int
    num_draft_tokens: int
    num_spec_tokens: int
    # Ordered per-verify-step arrays; populated only at the `detailed` level.
    per_step_accepted: list[int] | None = None
    per_step_drafted: list[int] | None = None


class PerRequestMetrics(OpenAIResponseModel):
    time_to_first_token_ms: float | None = None
    generation_time_ms: float | None = None
    queue_time_ms: float | None = None
    mean_itl_ms: float | None = None
    tokens_per_second: float | None = None
    # Experimental, subject to change.
    speculative_decoding: SpeculativeDecodingMetrics | None = None


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: UsageInfo | None = None


class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: str | None = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class LegacyStructuralTag(OpenAIBaseModel):
    begin: str
    # schema is the field, but that causes conflicts with pydantic so
    # instead use structural_tag_schema with an alias
    structural_tag_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    end: str


class LegacyStructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    structures: list[LegacyStructuralTag]
    triggers: list[str]


class StructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    format: Any


AnyStructuralTagResponseFormat: TypeAlias = (
    LegacyStructuralTagResponseFormat | StructuralTagResponseFormat
)


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object", or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | None = None


AnyResponseFormat: TypeAlias = (
    ResponseFormat | StructuralTagResponseFormat | LegacyStructuralTagResponseFormat
)


def structured_outputs_from_response_format(
    structured_outputs: StructuredOutputsParams | None,
    response_format: AnyResponseFormat | None,
) -> StructuredOutputsParams | None:
    """Apply ``response_format`` overrides to ``structured_outputs``."""
    if response_format is None or response_format.type == "text":
        return structured_outputs

    overrides: dict[str, Any]
    if response_format.type == "json_object":
        overrides = {"json_object": True}
    elif response_format.type == "json_schema":
        json_schema = response_format.json_schema
        assert json_schema is not None
        overrides = {"json": json_schema.json_schema}
    else:
        assert isinstance(
            response_format,
            (
                LegacyStructuralTagResponseFormat,
                StructuralTagResponseFormat,
            ),
        )
        overrides = {
            "structural_tag": json.dumps(response_format.model_dump(by_alias=True))
        }

    if structured_outputs is None:
        return StructuredOutputsParams(**overrides)

    return replace(structured_outputs, **overrides)


def validate_structural_tag_response_format(
    response_format: AnyStructuralTagResponseFormat | dict[str, Any],
) -> None:
    """Validate structural tags before they are sent to the engine.

    Engine-side validation reports malformed structural tags as generation
    failures. OpenAI request parsing should classify them as bad requests.
    """
    from pydantic import TypeAdapter, ValidationError

    if isinstance(response_format, dict):
        try:
            response_format = TypeAdapter(
                AnyStructuralTagResponseFormat
            ).validate_python(response_format)
        except ValidationError as exc:
            raise VLLMValidationError(
                "Invalid response_format structural_tag specification.",
                parameter="response_format",
            ) from exc

    try:
        payload = json.dumps(response_format.model_dump(by_alias=True))
        validate_structural_tag_payload(payload, parameter="response_format")
    except (TypeError, ValueError) as exc:
        raise VLLMValidationError(
            "Invalid response_format structural_tag specification.",
            parameter="response_format",
        ) from exc


def validate_structural_tag_payload(payload: Any, *, parameter: str) -> None:
    from vllm.sampling_params import (
        SamplingParams,
        StructuredOutputsParams,
        check_json_nesting,
    )
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

    if isinstance(payload, str) and not payload:
        raise VLLMValidationError(
            f"Invalid {parameter} structural_tag specification.",
            parameter=parameter,
        )

    if isinstance(payload, str):
        # Raised here so the error is not reported as a malformed tag below
        check_json_nesting(payload, structural_tag=True)
    try:
        validate_xgrammar_grammar(
            SamplingParams(
                structured_outputs=StructuredOutputsParams(structural_tag=payload)
            )
        )
    except (TypeError, ValueError, VLLMValidationError) as exc:
        raise VLLMValidationError(
            f"Invalid {parameter} structural_tag specification.",
            parameter=parameter,
        ) from exc


def validate_structured_outputs_structural_tag(
    structured_outputs: Any,
) -> None:
    from vllm.sampling_params import StructuredOutputsParams

    if isinstance(structured_outputs, StructuredOutputsParams):
        structural_tag = structured_outputs.structural_tag
    elif isinstance(structured_outputs, dict):
        structural_tag = structured_outputs.get("structural_tag")
    else:
        return
    if structural_tag is not None:
        validate_structural_tag_payload(
            structural_tag,
            parameter="structured_outputs",
        )


class StreamOptions(OpenAIBaseModel):
    include_usage: bool | None = False
    continuous_usage_stats: bool | None = False


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None
    defer_loading: bool | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        data = {k: v for k, v in data.items() if k in type(self).model_fields}
        if self.strict is None:
            data.pop("strict", None)
        if self.defer_loading is None:
            data.pop("defer_loading", None)
        return data


class FunctionCall(OpenAIBaseModel):
    # Internal field to preserve native tool call ID from tool parser.
    # Excluded from serialization to maintain OpenAI API compatibility
    # (function object should only contain 'name' and 'arguments').
    id: str | None = Field(default=None, exclude=True)
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=make_tool_call_id)
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: str | None = None
    type: Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: list[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: str | None = None


class DeltaMessage(OpenAIResponseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if len(data.get("tool_calls", [])) == 0:
            data.pop("tool_calls", None)
        return data


_FAST_INT64_MIN = -(2**63)
_FAST_INT64_MAX = 2**63 - 1


def build_fast_model_template(
    model_cls: type[BaseModel],
) -> tuple[dict[str, Any], tuple[tuple[str, Any], ...], dict[str, Any] | None]:
    """Snapshot static field defaults, default factories, and private attrs once."""
    base_defaults: dict[str, Any] = {}
    factories: list[tuple[str, Any]] = []
    for name, finfo in model_cls.model_fields.items():
        if finfo.default_factory is not None:
            factories.append((name, finfo.default_factory))
        elif finfo.default is not PydanticUndefined:
            base_defaults[name] = finfo.default
    priv_attrs = getattr(model_cls, "__private_attributes__", None)
    priv_template = (
        {k: attr.default for k, attr in priv_attrs.items()}
        if priv_attrs
        else None
    )
    return base_defaults, tuple(factories), priv_template


def fast_validate_scalar_fields(
    data: dict[str, Any],
    field_kinds: dict[str, str],
    base_defaults: dict[str, Any],
    factories: tuple[tuple[str, Any], ...],
) -> dict[str, Any] | None:
    """Validate scalar fields in ``data`` against ``field_kinds``.

    Returns populated field dict ready for ``model_construct`` or ``None`` if any
    key or value requires full Pydantic validation.
    """
    values = base_defaults.copy()
    for name, fn in factories:
        if name not in data:
            values[name] = fn()
    for key, val in data.items():
        kind = field_kinds.get(key)
        if kind is None:
            return None
        if kind == "skip":
            values[key] = val
        elif kind == "str_or_none":
            if val is not None and type(val) is not str:
                return None
            values[key] = val
        elif kind == "str":
            if type(val) is not str:
                return None
            values[key] = val
        elif kind == "bool":
            if type(val) is not bool:
                return None
            values[key] = val
        elif kind == "bool_or_none":
            if val is not None and type(val) is not bool:
                return None
            values[key] = val
        elif kind == "int":
            if type(val) is not int:
                return None
            values[key] = val
        elif kind == "int_or_none":
            if val is not None and type(val) is not int:
                return None
            values[key] = val
        elif kind == "ge1_int_or_none":
            if val is not None and (type(val) is not int or val < 1):
                return None
            values[key] = val
        elif kind == "int64_or_none":
            if val is not None and (
                type(val) is not int
                or not (_FAST_INT64_MIN <= val <= _FAST_INT64_MAX)
            ):
                return None
            values[key] = val
        elif kind == "int64":
            if type(val) is not int or not (
                _FAST_INT64_MIN <= val <= _FAST_INT64_MAX
            ):
                return None
            values[key] = val
        elif kind == "float":
            if type(val) is float:
                values[key] = val
            elif type(val) is int:
                values[key] = float(val)
            else:
                return None
        elif kind == "float_or_none":
            if val is None or type(val) is float:
                values[key] = val
            elif type(val) is int:
                values[key] = float(val)
            else:
                return None
        elif kind == "stop":
            if val is None or type(val) is str:
                values[key] = val
            elif (
                type(val) is list
                and len(val) <= envs.VLLM_MAX_STOP_STRINGS
                and all(type(s) is str for s in val)
            ):
                values[key] = val
            else:
                return None
        elif kind == "false_or_none":
            if val is not None and val is not False:
                return None
            values[key] = val
        elif kind == "zero_or_none":
            if val is not None and (type(val) is not int or val != 0):
                return None
            values[key] = 0
        elif kind == "tool_choice_none":
            if val is not None and val != "none":
                return None
            values[key] = val
        elif kind == "ct_kwargs":
            if val is None:
                values[key] = None
            elif (
                type(val) is dict
                and "chat_template" not in val
                and all(
                    type(ck) is str
                    and (cv is None or type(cv) in (bool, int, float, str))
                    for ck, cv in val.items()
                )
            ):
                values[key] = val
            else:
                return None
        elif kind == "trunc_side":
            if val is not None and val != "left" and val != "right":
                return None
            values[key] = val
        elif kind == "trunc_toks":
            if val is not None and (
                type(val) is not int or val < -1 or val > _FAST_INT64_MAX
            ):
                return None
            values[key] = val
        elif kind == "cache_salt":
            if val is not None and (
                type(val) is not str
                or not val
                or len(val) > _MAX_CACHE_SALT_LENGTH
                or "@" in val
                or "/" in val
                or "\\" in val
                or "\x00" in val
            ):
                return None
            values[key] = val
        else:
            return None
    return values

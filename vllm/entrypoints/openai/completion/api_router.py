# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request, Response

from vllm.entrypoints.serve.instrumentator._fast_middleware import (
    FastStreamingResponse,
)
from vllm.entrypoints.serve.utils.orjson_route import FastOrjsonAPIRoute
from fastapi.responses import JSONResponse
from vllm.entrypoints.serve.utils.api_utils import create_model_json_response

from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    json_or_sse_responses,
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.entrypoints.serve.utils.sse_keep_alive import with_sse_keep_alive
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter(route_class=FastOrjsonAPIRoute)
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def completion(request: Request) -> OpenAIServingCompletion | None:
    return request.app.state.openai_serving_completion


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    **json_or_sse_responses(
        CompletionResponse,
        CompletionStreamResponse,
        {
            HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
            HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        },
    ),
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = completion(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Completions API")

    handler_cls = type(handler)
    if (
        getattr(handler_cls, "create_completion", None)
        is OpenAIServingCompletion.create_completion
    ):
        use_raw_json = (
            not request.stream
            and getattr(handler_cls, "request_output_to_completion_response", None)
            is OpenAIServingCompletion.request_output_to_completion_response
        )
        generator = await handler.create_completion(
            request, raw_request, _yield_bytes=True, _raw_json=use_raw_json
        )
    else:
        generator = await handler.create_completion(request, raw_request)

    if type(generator) is bytes:
        return Response(
            content=generator,
            status_code=HTTPStatus.OK,
            headers=metrics_header(metrics_header_format),
            media_type="application/json",
        )

    if isinstance(generator, ErrorResponse):
        return create_model_json_response(
            generator, status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        return create_model_json_response(
            generator, headers=metrics_header(metrics_header_format)
        )

    args = getattr(raw_request.app.state, "args", None)
    keep_alive_interval = getattr(args, "sse_keep_alive_interval", 0)
    return FastStreamingResponse(
        content=(
            with_sse_keep_alive(generator, float(keep_alive_interval))
            if keep_alive_interval
            else generator
        ),
        media_type="text/event-stream",
    )


def attach_router(app: FastAPI):
    app.include_router(router)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast APIRoute bypassing redundant FastAPI/Starlette JSON body parsing."""

from collections.abc import Callable, Coroutine
from contextlib import AsyncExitStack
import inspect
from typing import Any

import orjson
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute, serialize_response
from pydantic import BaseModel, ValidationError
from starlette._exception_handler import _lookup_exception_handler
from starlette._utils import is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _is_validate_json_dep(sub_dep: Any) -> bool:
    call = getattr(sub_dep, "call", None)
    if call is None:
        return False
    if getattr(call, "__name__", None) != "validate_json_request":
        return False
    return (
        not sub_dep.dependencies
        and not sub_dep.path_params
        and not sub_dep.query_params
        and not sub_dep.header_params
        and not sub_dep.cookie_params
        and not sub_dep.body_params
        and not sub_dep._uses_scopes
    )


def _fast_request_response(
    func: Callable[[Request], Coroutine[Any, Any, Response]],
) -> ASGIApp:
    """ASGI wrapper replacing dual AsyncExitStack + async sender on fast routes."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive, send)
        exc_handlers = scope.get("starlette.exception_handlers")
        if exc_handlers is None:
            exception_handlers: dict[Any, Any] = {}
            status_handlers: dict[int, Any] = {}
        else:
            exception_handlers, status_handlers = exc_handlers

        response_started = False

        def sender_fast(message: Message) -> Any:
            nonlocal response_started
            if not response_started and message["type"] == "http.response.start":
                response_started = True
            return send(message)

        sender_fast._fast_post_start_send = send  # type: ignore[attr-defined]

        try:
            response = await func(request)
            await response(scope, receive, sender_fast)
        except Exception as exc:
            handler = None
            if isinstance(exc, HTTPException):
                handler = status_handlers.get(exc.status_code)
            if handler is None:
                handler = _lookup_exception_handler(exception_handlers, exc)
            if handler is None:
                raise exc
            if response_started:
                raise RuntimeError(
                    "Caught handled exception, but response already started."
                ) from exc
            if is_async_callable(handler):
                err_response = await handler(request, exc)  # type: ignore[arg-type]
            else:
                err_response = await run_in_threadpool(handler, request, exc)  # type: ignore[arg-type]
            if err_response is not None:
                await err_response(scope, receive, sender_fast)

    return app


class FastOrjsonAPIRoute(APIRoute):
    """APIRoute specialization for single JSON-model POST handlers.

    FastAPI's default ``APIRoute`` parses ``content-type`` with
    ``email.message.Message()``, decodes JSON into dicts via stdlib ``json`` in
    ``request.json()``, runs ``solve_dependencies()`` with an ``AsyncExitStack``,
    and validates through ``ModelField`` / ``TypeAdapter``. When a route takes
    one ``BaseModel`` body plus optional ``Request`` and (optionally)
    ``validate_json_request``, we parse and validate the raw request bytes once
    and fall back to the stock handler on any validation/header edge case.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._fast_json_handler: Callable[[Request], Coroutine[Any, Any, Response]] | None = None
        super().__init__(*args, **kwargs)
        if getattr(self, "_fast_orjson_active", False) and self._fast_json_handler is not None:
            self.app = _fast_request_response(self._fast_json_handler)

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        self._fast_orjson_active = False
        orig_handler = super().get_route_handler()
        dep = self.dependant

        if (
            dep.call is None
            or not inspect.iscoroutinefunction(dep.call)
            or self._embed_body_fields
            or dep.is_async_gen_callable
            or dep.is_gen_callable
            or len(dep.body_params) != 1
            or dep.path_params
            or dep.query_params
            or dep.header_params
            or dep.cookie_params
            or dep._uses_scopes
            or dep.own_oauth_scopes
            or dep.parent_oauth_scopes
            or dep.websocket_param_name is not None
            or dep.http_connection_param_name is not None
            or dep.response_param_name is not None
            or dep.background_tasks_param_name is not None
            or dep.security_scopes_param_name is not None
        ):
            return orig_handler

        deps = dep.dependencies
        if len(deps) > 1 or (len(deps) == 1 and not _is_validate_json_dep(deps[0])):
            return orig_handler

        body_param = dep.body_params[0]
        body_cls = body_param.field_info.annotation
        if not (isinstance(body_cls, type) and issubclass(body_cls, BaseModel)):
            return orig_handler
        if body_param.field_info.metadata:
            return orig_handler

        self._fast_orjson_active = True

        endpoint = dep.call
        body_name = body_param.name
        req_name = dep.request_param_name
        has_json_check = len(deps) == 1
        actual_status = self.status_code if self.status_code is not None else 200
        model_validate = body_cls.model_validate
        fast_validate = getattr(body_cls, "_try_fast_validate", None)
        orjson_loads = orjson.loads
        orjson_decode_err = orjson.JSONDecodeError
        response_field = self.response_field
        resp_include = self.response_model_include
        resp_exclude = self.response_model_exclude
        resp_by_alias = self.response_model_by_alias
        resp_exclude_unset = self.response_model_exclude_unset
        resp_exclude_defaults = self.response_model_exclude_defaults
        resp_exclude_none = self.response_model_exclude_none

        async def _call_orig(request: Request) -> Response:
            async with AsyncExitStack() as mw_stack:
                if "fastapi_middleware_astack" not in request.scope:
                    request.scope["fastapi_middleware_astack"] = mw_stack
                if (
                    "fastapi_inner_astack" in request.scope
                    and "fastapi_function_astack" in request.scope
                ):
                    return await orig_handler(request)
                async with AsyncExitStack() as request_stack:
                    request.scope["fastapi_inner_astack"] = request_stack
                    async with AsyncExitStack() as function_stack:
                        request.scope["fastapi_function_astack"] = function_stack
                        return await orig_handler(request)

        async def fast_json_handler(request: Request) -> Response:
            if has_json_check:
                ct = request.headers.get("content-type")
                if ct != "application/json":
                    if ct is None:
                        return await _call_orig(request)
                    if ct.partition(";")[0].strip().lower() != "application/json":
                        return await _call_orig(request)

            raw_body = await request.body()
            if not raw_body:
                return await _call_orig(request)

            try:
                parsed_json = orjson_loads(raw_body)
                if type(parsed_json) is not dict:
                    return await _call_orig(request)
                request._json = parsed_json
                body_obj = (
                    fast_validate(parsed_json) if fast_validate is not None else None
                )
                if body_obj is None:
                    body_obj = model_validate(parsed_json)
            except (orjson_decode_err, ValidationError):
                return await _call_orig(request)

            if req_name is not None:
                result = await endpoint(**{body_name: body_obj, req_name: request})
            else:
                result = await endpoint(**{body_name: body_obj})

            if isinstance(result, Response):
                return result
            encoded = await serialize_response(
                field=response_field,
                response_content=result,
                include=resp_include,
                exclude=resp_exclude,
                by_alias=resp_by_alias,
                exclude_unset=resp_exclude_unset,
                exclude_defaults=resp_exclude_defaults,
                exclude_none=resp_exclude_none,
                is_coroutine=True,
            )
            return JSONResponse(content=encoded, status_code=actual_status)

        self._fast_json_handler = fast_json_handler
        return fast_json_handler

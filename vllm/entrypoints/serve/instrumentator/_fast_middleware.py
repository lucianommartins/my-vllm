# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead FastAPI Prometheus instrumentator middleware for vLLM serving."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from http import HTTPStatus
from timeit import default_timer
import types
from typing import Any

from fastapi.middleware.asyncexitstack import AsyncExitStackMiddleware
from fastapi.routing import APIRoute
from prometheus_client import Counter, Histogram, Summary
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.middleware import (
    PrometheusInstrumentatorMiddleware,
)
from starlette._exception_handler import _lookup_exception_handler
from starlette._utils import is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.middleware.exceptions import ExceptionMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Match, Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocket

_ROUTE_CACHE_MAX = 4096
_LABEL_CACHE_MAX = 4096
_STATUS_GROUP_BY_CODE = ("0xx", "1xx", "2xx", "3xx", "4xx", "5xx")


def _extract_content_length(raw_headers: Any) -> int:
    for key, val in raw_headers:
        if key == b"content-length" or (
            len(key) == 14 and key[0] in (99, 67) and key.lower() == b"content-length"
        ):
            return int(val)
    return 0


class FastPrometheusMiddleware(PrometheusInstrumentatorMiddleware):
    """Drop-in subclass of PrometheusInstrumentatorMiddleware.

    Eliminates per-request full route-table scans, per-chunk ``async def``
    send wrappers, and per-request ``Request`` / ``Headers`` / ``Response`` /
    ``Info`` / ``asyncio.gather()`` allocations on the default metrics path
    while preserving byte-exact Prometheus metric semantics.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        metric_namespace: str = "",
        metric_subsystem: str = "",
        should_only_respect_2xx_for_highr: bool = False,
        should_exclude_streaming_duration: bool = False,
        custom_labels: dict[Any, Any] | None = None,
        instrumentations: Any = (),
        **kwargs: Any,
    ) -> None:
        custom_labels_arg = {} if custom_labels is None else custom_labels
        self._has_custom_instrumentations = bool(instrumentations)
        super().__init__(
            app,
            metric_namespace=metric_namespace,
            metric_subsystem=metric_subsystem,
            should_only_respect_2xx_for_highr=should_only_respect_2xx_for_highr,
            should_exclude_streaming_duration=should_exclude_streaming_duration,
            custom_labels=custom_labels_arg,
            instrumentations=instrumentations,
            **kwargs,
        )
        self._route_cache: dict[
            tuple[str, str, str], tuple[str, bool, bool, bool]
        ] = {}
        self._cached_routes_obj: Any = None
        self._cached_routes_sig: tuple[int, Any, Any] = (-1, None, None)

        self._label_cache: dict[
            tuple[str, str, str], tuple[Any, Any, Any, Any]
        ] = {}
        self._use_fast_default = False
        self._m_req_total: Counter | None = None
        self._m_req_size: Summary | None = None
        self._m_resp_size: Summary | None = None
        self._m_dur_highr: Histogram | None = None
        self._m_dur_lowr: Histogram | None = None
        self._should_exclude_streaming_duration = should_exclude_streaming_duration

        if (
            not self._has_custom_instrumentations
            and not self.async_instrumentations
            and not self.body_handlers
            and not self.should_instrument_requests_inprogress
            and not self.should_respect_env_var
            and not metric_namespace
            and not metric_subsystem
            and not custom_labels_arg
            and not should_only_respect_2xx_for_highr
            and not should_exclude_streaming_duration
            and len(self.instrumentations) == 1
        ):
            self._try_bind_default_metrics()

    def _try_bind_default_metrics(self) -> None:
        names_map = getattr(self.registry, "_names_to_collectors", None)
        if not isinstance(names_map, dict):
            return
        req_total = names_map.get("http_requests_total") or names_map.get(
            "http_requests"
        )
        req_size = names_map.get("http_request_size_bytes")
        resp_size = names_map.get("http_response_size_bytes")
        dur_highr = names_map.get("http_request_duration_highr_seconds")
        dur_lowr = names_map.get("http_request_duration_seconds")
        if (
            isinstance(req_total, Counter)
            and req_total._labelnames == ("method", "status", "handler")
            and isinstance(req_size, Summary)
            and req_size._labelnames == ("handler",)
            and isinstance(resp_size, Summary)
            and resp_size._labelnames == ("handler",)
            and isinstance(dur_highr, Histogram)
            and dur_highr._labelnames == ()
            and isinstance(dur_lowr, Histogram)
            and dur_lowr._labelnames == ("method", "handler")
        ):
            self._m_req_total = req_total
            self._m_req_size = req_size
            self._m_resp_size = resp_size
            self._m_dur_highr = dur_highr
            self._m_dur_lowr = dur_lowr
            self._use_fast_default = True

    def _resolve_route(
        self, scope: Scope, method: str
    ) -> tuple[str, bool, bool, bool]:
        app = scope.get("app")
        routes = app.routes if app is not None else ()
        n_routes = len(routes)
        sig = (
            n_routes,
            routes[0] if n_routes else None,
            routes[-1] if n_routes else None,
        )
        if routes is not self._cached_routes_obj or sig != self._cached_routes_sig:
            self._cached_routes_obj = routes
            self._cached_routes_sig = sig
            self._route_cache.clear()

        path = scope.get("path", "")
        root_path = scope.get("root_path", "")
        key = (method, path, root_path)
        cached = self._route_cache.get(key)
        if cached is not None:
            return cached

        request = Request(scope)
        raw_handler, is_templated = self._get_handler(request)
        is_excluded = self._is_handler_excluded(raw_handler, is_templated)
        handler = (
            "none" if not is_templated and self.should_group_untemplated else raw_handler
        )
        needs_body = (
            bool(self.body_handlers)
            and any(pattern.search(handler) for pattern in self.body_handlers)
        )
        entry = (handler, is_templated, is_excluded, needs_body)
        if len(self._route_cache) < _ROUTE_CACHE_MAX:
            self._route_cache[key] = entry
        return entry

    def _get_metric_children(
        self, method: str, status: str, handler: str
    ) -> tuple[Any, Any, Any, Any]:
        key = (method, status, handler)
        cached = self._label_cache.get(key)
        if cached is not None:
            return cached
        assert self._m_req_total is not None
        assert self._m_req_size is not None
        assert self._m_resp_size is not None
        assert self._m_dur_lowr is not None
        children = (
            self._m_req_total.labels(method, status, handler),
            self._m_req_size.labels(handler),
            self._m_resp_size.labels(handler),
            self._m_dur_lowr.labels(method, handler),
        )
        if len(self._label_cache) < _LABEL_CACHE_MAX:
            self._label_cache[key] = children
        return children

    def _format_status(self, status_code: Any) -> str:
        if (
            self.should_group_status_codes
            and type(status_code) is int
            and 100 <= status_code < 600
        ):
            return _STATUS_GROUP_BY_CODE[status_code // 100]
        status = (
            str(status_code.value)
            if isinstance(status_code, HTTPStatus)
            else str(status_code)
        )
        if self.should_group_status_codes:
            return status[0] + "xx"
        return status

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        method = scope.get("method", "")
        handler, _is_templated, is_excluded, needs_body = self._resolve_route(
            scope, method
        )

        if is_excluded:
            return await self.app(scope, receive, send)

        start_time = default_timer()
        status_code: Any = 500
        headers: Any = ()

        if self._use_fast_default:
            def send_wrapper_fast(message: Message) -> Any:
                if message["type"] == "http.response.start":
                    nonlocal status_code, headers
                    headers = message["headers"]
                    status_code = message["status"]
                return send(message)

            send_wrapper_fast._fast_post_start_send = send  # type: ignore[attr-defined]

            try:
                await self.app(scope, receive, send_wrapper_fast)
            finally:
                duration = max(default_timer() - start_time, 0.0)
                if self.should_round_latency_decimals:
                    duration = round(duration, self.round_latency_decimals)
                status = self._format_status(status_code)
                req_len = _extract_content_length(scope.get("headers", ()))
                resp_len = _extract_content_length(headers)
                cnt_child, req_sz_child, resp_sz_child, lowr_child = (
                    self._get_metric_children(method, status, handler)
                )
                cnt_child.inc()
                req_sz_child.observe(req_len)
                resp_sz_child.observe(resp_len)
                assert self._m_dur_highr is not None
                self._m_dur_highr.observe(duration)
                lowr_child.observe(duration)
            return

        request = Request(scope)
        inprogress: Any = None
        if self.inprogress:
            if self.inprogress_labels:
                inprogress = self.inprogress.labels(method, handler)
            else:
                inprogress = self.inprogress
            inprogress.inc()

        body = b""
        response_start_time: float | None = None

        if needs_body:
            def send_wrapper(message: Message) -> Any:
                msg_type = message["type"]
                if msg_type == "http.response.start":
                    nonlocal status_code, headers, response_start_time
                    headers = message["headers"]
                    status_code = message["status"]
                    response_start_time = default_timer()
                elif msg_type == "http.response.body" and message["body"]:
                    nonlocal body
                    body += message["body"]
                return send(message)
        else:
            def send_wrapper(message: Message) -> Any:
                if message["type"] == "http.response.start":
                    nonlocal status_code, headers, response_start_time
                    headers = message["headers"]
                    status_code = message["status"]
                    response_start_time = default_timer()
                return send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = max(default_timer() - start_time, 0.0)
            duration_without_streaming = (
                max(response_start_time - start_time, 0.0)
                if response_start_time
                else 0.0
            )
            if self.should_instrument_requests_inprogress and inprogress is not None:
                inprogress.dec()
            if self.should_round_latency_decimals:
                duration = round(duration, self.round_latency_decimals)
                duration_without_streaming = round(
                    duration_without_streaming, self.round_latency_decimals
                )
            status = self._format_status(status_code)
            response = Response(
                content=body, headers=Headers(raw=headers), status_code=status_code
            )
            info = metrics.Info(
                request=request,
                response=response,
                method=method,
                modified_handler=handler,
                modified_status=status,
                modified_duration=duration,
                modified_duration_without_streaming=duration_without_streaming,
            )
            for instrumentation in self.instrumentations:
                instrumentation(info)
            if self.async_instrumentations:
                await asyncio.gather(
                    *[
                        instrumentation(info)
                        for instrumentation in self.async_instrumentations
                    ]
                )


class FastInstrumentator(Instrumentator):
    """Instrumentator subclass installing :class:`FastPrometheusMiddleware`."""

    def instrument(
        self,
        app: Any,
        metric_namespace: str = "",
        metric_subsystem: str = "",
        should_only_respect_2xx_for_highr: bool = False,
        latency_highr_buckets: Any = (
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1,
            1.5,
            2,
            2.5,
            3,
            3.5,
            4,
            4.5,
            5,
            7.5,
            10,
            30,
            60,
        ),
        latency_lowr_buckets: Any = (0.1, 0.5, 1),
    ) -> "FastInstrumentator":
        if self.should_respect_env_var and not self._should_instrumentate():
            return self

        app.add_middleware(
            FastPrometheusMiddleware,
            should_group_status_codes=self.should_group_status_codes,
            should_ignore_untemplated=self.should_ignore_untemplated,
            should_group_untemplated=self.should_group_untemplated,
            should_round_latency_decimals=self.should_round_latency_decimals,
            should_respect_env_var=self.should_respect_env_var,
            should_instrument_requests_inprogress=self.should_instrument_requests_inprogress,
            should_exclude_streaming_duration=self.should_exclude_streaming_duration,
            round_latency_decimals=self.round_latency_decimals,
            env_var_name=self.env_var_name,
            inprogress_name=self.inprogress_name,
            inprogress_labels=self.inprogress_labels,
            instrumentations=self.instrumentations,
            async_instrumentations=self.async_instrumentations,
            excluded_handlers=self.excluded_handlers,
            body_handlers=self.body_handlers,
            metric_namespace=metric_namespace,
            metric_subsystem=metric_subsystem,
            should_only_respect_2xx_for_highr=should_only_respect_2xx_for_highr,
            latency_highr_buckets=latency_highr_buckets,
            latency_lowr_buckets=latency_lowr_buckets,
            registry=self.registry,
        )
        return self


class FastServerErrorMiddleware(ServerErrorMiddleware):
    """Drop-in ServerErrorMiddleware without an async send wrapper per chunk."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        def _send_fast(message: Message) -> Any:
            nonlocal response_started
            if not response_started and message["type"] == "http.response.start":
                response_started = True
            return send(message)

        try:
            await self.app(scope, receive, _send_fast)
        except Exception as exc:
            request = Request(scope)
            if self.debug:
                response = self.debug_response(request, exc)
            elif self.handler is None:
                response = self.error_response(request, exc)
            else:
                if is_async_callable(self.handler):
                    response = await self.handler(request, exc)  # type: ignore[assignment, arg-type]
                else:
                    response = await run_in_threadpool(self.handler, request, exc)  # type: ignore[arg-type]

            if not response_started:
                await response(scope, receive, send)

            raise exc


class FastExceptionMiddleware(ExceptionMiddleware):
    """Drop-in ExceptionMiddleware without per-chunk async send wrappers."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        scope["starlette.exception_handlers"] = (
            self._exception_handlers,
            self._status_handlers,
        )
        response_started = False

        def sender_fast(message: Message) -> Any:
            nonlocal response_started
            if not response_started and message["type"] == "http.response.start":
                response_started = True
            return send(message)

        sender_fast._fast_post_start_send = send  # type: ignore[attr-defined]

        try:
            await self.app(scope, receive, sender_fast)
        except Exception as exc:
            handler = None
            if isinstance(exc, HTTPException):
                handler = self._status_handlers.get(exc.status_code)
            if handler is None:
                handler = _lookup_exception_handler(self._exception_handlers, exc)
            if handler is None:
                raise exc
            if response_started:
                raise RuntimeError(
                    "Caught handled exception, but response already started."
                ) from exc

            conn: Request | WebSocket = (
                Request(scope, receive, send)
                if scope_type == "http"
                else WebSocket(scope, receive, send)
            )
            if is_async_callable(handler):
                response = await handler(conn, exc)  # type: ignore[arg-type]
            else:
                response = await run_in_threadpool(handler, conn, exc)  # type: ignore[arg-type]
            if response is not None:
                await response(scope, receive, sender_fast)


class FastCORSMiddleware(CORSMiddleware):
    """Drop-in CORSMiddleware skipping Headers(scope) when no Origin header exists."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for key, _ in scope.get("headers", ()):
            if key == b"origin" or (
                len(key) == 6 and key[0] in (111, 79) and key.lower() == b"origin"
            ):
                await super().__call__(scope, receive, send)
                return

        await self.app(scope, receive, send)


class FastStreamingResponse(StreamingResponse):
    """StreamingResponse avoiding per-chunk `bytes | memoryview` Union and wrapper overhead."""

    async def stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        body_send = send
        nxt = getattr(body_send, "_fast_post_start_send", None)
        while nxt is not None:
            body_send = nxt
            nxt = getattr(body_send, "_fast_post_start_send", None)

        charset = self.charset
        async for chunk in self.body_iterator:
            if type(chunk) is not bytes and not isinstance(chunk, (bytes, memoryview)):
                chunk = chunk.encode(charset)
            await body_send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": True,
                }
            )
        await body_send({"type": "http.response.body", "body": b"", "more_body": False})


class FastAsyncExitStackMiddleware(AsyncExitStackMiddleware):
    """Route static HTTP paths in O(1) and skip AsyncExitStack for FastOrjsonAPIRoute."""

    def __init__(self, app: ASGIApp, router: Any) -> None:
        super().__init__(app)
        self._router = router
        self._routes_snapshot: list[Any] | None = None
        # (method, path) -> (route_app, route_endpoint, api_route_or_none, skip_exit_stack)
        self._static_routes: dict[
            tuple[str, str], tuple[ASGIApp, Any, Any, bool]
        ] = {}

    def _rebuild_static_routes(self, routes: list[Any]) -> None:
        static_routes: dict[tuple[str, str], tuple[ASGIApp, Any, Any, bool]] = {}
        earlier_dynamic: list[Any] = []

        for r in routes:
            matches_fn = getattr(type(r), "matches", None)
            if (
                isinstance(r, Route)
                and (matches_fn is Route.matches or matches_fn is APIRoute.matches)
                and not r.param_convertors
                and r.methods
            ):
                skip_exit_stack = bool(getattr(r, "_fast_orjson_active", False))
                route_obj = r if matches_fn is APIRoute.matches else None
                entry = (r.app, r.endpoint, route_obj, skip_exit_stack)
                path = r.path
                for method in r.methods:
                    key = (method, path)
                    if key in static_routes:
                        continue
                    blocked = False
                    if earlier_dynamic:
                        probe_scope: Scope = {
                            "type": "http",
                            "method": method,
                            "path": path,
                            "root_path": "",
                            "headers": [],
                        }
                        for prev_r in earlier_dynamic:
                            try:
                                match, _ = prev_r.matches(probe_scope)
                            except Exception:
                                blocked = True
                                break
                            if match == Match.FULL or (
                                not isinstance(prev_r, Route) and match != Match.NONE
                            ):
                                blocked = True
                                break
                    if not blocked:
                        static_routes[key] = entry
            else:
                earlier_dynamic.append(r)

        self._static_routes = static_routes
        self._routes_snapshot = list(routes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and not scope.get("root_path")
            and not scope.get("path_params")
        ):
            router = self._router
            routes = router.routes
            snapshot = self._routes_snapshot
            if snapshot is None or len(routes) != len(snapshot) or routes != snapshot:
                self._rebuild_static_routes(routes)

            entry = self._static_routes.get((scope["method"], scope["path"]))
            if entry is not None:
                route_app, route_endpoint, route_obj, skip_exit_stack = entry
                if "router" not in scope:
                    scope["router"] = router
                scope["endpoint"] = route_endpoint
                scope["path_params"] = {}
                if route_obj is not None:
                    scope["route"] = route_obj
                if skip_exit_stack:
                    await route_app(scope, receive, send)
                    return
                async with AsyncExitStack() as stack:
                    scope[self.context_name] = stack
                    await route_app(scope, receive, send)
                return

        async with AsyncExitStack() as stack:
            scope[self.context_name] = stack
            await self.app(scope, receive, send)


def install_fast_middleware_stack(app: Any) -> None:
    """Install low-overhead error/CORS middleware wrappers on a FastAPI app."""
    if not (
        hasattr(app, "build_middleware_stack")
        and hasattr(app, "user_middleware")
        and hasattr(app, "exception_handlers")
        and hasattr(app, "router")
    ):
        return

    def _fast_build_middleware_stack(self: Any) -> ASGIApp:
        debug = self.debug
        error_handler = None
        exception_handlers: dict[Any, Any] = {}

        for key, value in self.exception_handlers.items():
            if key in (500, Exception):
                error_handler = value
            else:
                exception_handlers[key] = value

        user_middleware = [
            Middleware(FastCORSMiddleware, *m.args, **m.kwargs)
            if m.cls is CORSMiddleware
            else m
            for m in self.user_middleware
        ]

        middleware = (
            [Middleware(FastServerErrorMiddleware, handler=error_handler, debug=debug)]
            + user_middleware
            + [
                Middleware(
                    FastExceptionMiddleware,
                    handlers=exception_handlers,
                    debug=debug,
                ),
                Middleware(FastAsyncExitStackMiddleware, router=self.router),
            ]
        )

        inner_app = self.router
        for cls, args, kwargs in reversed(middleware):
            inner_app = cls(inner_app, *args, **kwargs)
        return inner_app

    app.build_middleware_stack = types.MethodType(_fast_build_middleware_stack, app)

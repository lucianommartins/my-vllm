# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from collections.abc import Awaitable

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class XRequestIdMiddleware:
    """Middleware the set's the X-Request-Id header for each response
    to a random uuid4 (hex) value if the header isn't already
    present in the request, otherwise use the provided request id.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket"):
            return self.app(scope, receive, send)

        request_id_bytes: bytes | None = None
        headers = scope.get("headers")
        if headers:
            for k, v in headers:
                if k == b"x-request-id" or (
                    len(k) == 12 and k.lower() == b"x-request-id"
                ):
                    request_id_bytes = v
                    break

        def send_with_request_id(message: Message) -> Awaitable[None]:
            """Custom send function to mutate response start headers in-place."""
            if message["type"] == "http.response.start":
                rid = (
                    request_id_bytes
                    if request_id_bytes is not None
                    else uuid.uuid4().hex.encode("latin-1")
                )
                resp_headers = message.get("headers")
                if isinstance(resp_headers, list):
                    resp_headers.append((b"x-request-id", rid))
                else:
                    MutableHeaders(raw=resp_headers).append(
                        "X-Request-Id", rid.decode("latin-1")
                    )
            return send(message)

        return self.app(scope, receive, send_with_request_id)

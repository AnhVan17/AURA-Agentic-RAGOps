import time
from typing import Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.settings import APPSETTINGS

class RateLimiter(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.window = APPSETTINGS.api["rate_limit"]["window_sec"]
        self.max_req = APPSETTINGS.api["rate_limit"]["max_requests"]
        self.buckets: Dict[str, list] = {}  

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = int(time.time())
        reset, cnt = self.buckets.get(ip, [now + self.window, 0])

        if now > reset:
            reset, cnt = now + self.window, 0

        cnt += 1
        self.buckets[ip] = [reset, cnt]

        if cnt > self.max_req:
            retry_after = max(1, reset - now)
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try later.", "retry_after_sec": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)}
            )

        resp: Response = await call_next(request)
        resp.headers["X-RateLimit-Limit"] = str(self.max_req)
        resp.headers["X-RateLimit-Remaining"] = str(max(0, self.max_req - cnt))
        resp.headers["X-RateLimit-Reset"] = str(reset)
        return resp

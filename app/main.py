from fastapi import FastAPI
from app.settings import APPSETTINGS
from app.api.routes_healthz import router as health_router
from app.api.routes_toy import router as toy_router
from app.api.routes_session import router as session_router
from app.api.routes_ask import router as ask_router
from app.middleware_rate_limit import RateLimiter
from ops.observability import init_langsmith

# --- Ngày 1: Khởi tạo Observability ---
init_langsmith(project_name=APPSETTINGS.app.name)

app = FastAPI(title=APPSETTINGS.app.name, version=APPSETTINGS.app.version)
app.include_router(health_router)
app.include_router(toy_router)
app.include_router(session_router)
app.include_router(ask_router)
app.add_middleware(RateLimiter)
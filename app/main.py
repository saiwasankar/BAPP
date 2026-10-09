"""BAPPS FastAPI application entry point.

Run from the project root:  uvicorn app.main:app --reload
Interactive API docs:       http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .database import init_db
from .errors import BappsError
from .routers import admin, egate, immigration, passengers


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="BAPPS - Biometric Aviation Passenger Processing System",
    version="1.0.0",
    description=(
        "Backend prototype for automated e-gate passenger processing. Pipeline: "
        "1) scan documents, 2) liveness + face match (3 attempts max), "
        "3) immigration and watchlist checks, 4) gate authorisation or officer override, "
        "5) tamper-evident audit log. Face capture is simulated; there is no authentication "
        "in this prototype."
    ),
    lifespan=lifespan,
)

# Open CORS so the kiosk UI can be served from a different local port later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(BappsError)
async def bapps_error_handler(request: Request, exc: BappsError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


app.include_router(passengers.router)
app.include_router(egate.router)
app.include_router(immigration.router)
app.include_router(admin.router)


@app.get("/api/health", tags=["System"], summary="Liveness probe")
def health():
    return {"status": "ok", "service": "bapps-backend", "version": app.version}

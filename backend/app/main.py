"""FastAPI entrypoint for Insulin Matrix AI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_NAME, APP_VERSION, MEDICAL_DISCLAIMER
from app.routers.api import router
from app.state import STATE

logger = logging.getLogger("insulin_matrix")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Restore settings and optionally auto-load the synthetic sample.
    for key in list(STATE.settings.keys()):
        stored = STATE.db.get_setting(key, None)
        if stored is not None:
            STATE.settings[key] = stored
    if STATE.settings.get("auto_load_sample", True):
        try:
            from app.services import dataset_service

            dataset_service.load_default_sample()
            logger.info("Loaded default synthetic dataset.")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not auto-load sample dataset: %s", exc)
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=MEDICAL_DISCLAIMER,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
        "disclaimer": MEDICAL_DISCLAIMER,
    }

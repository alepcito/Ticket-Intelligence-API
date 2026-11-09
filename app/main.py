"""HTTP inference service. Load only locally trained, trusted artifacts."""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("uvicorn.error")
ROOT = Path(__file__).resolve().parents[1]


class Ticket(BaseModel):
    text: str = Field(min_length=5, max_length=5000, examples=["Me cobraron dos veces la suscripción"])

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value


class BatchRequest(BaseModel):
    tickets: list[Ticket] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    category: str
    confidence: float
    needs_review: bool
    probabilities: dict[str, float]
    model_version: str


def create_app(model_path: Path | None = None, threshold: float | None = None) -> FastAPI:
    path = model_path or Path(os.getenv("MODEL_PATH", str(ROOT / "artifacts/model.joblib")))
    review_threshold = threshold if threshold is not None else float(os.getenv("REVIEW_THRESHOLD", "0.60"))
    if not 0 <= review_threshold <= 1:
        raise ValueError("REVIEW_THRESHOLD must be between 0 and 1")

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        if not path.is_file():
            raise RuntimeError("Model missing. Run: python -m training.train")
        artifact = joblib.load(path)
        api.state.model = artifact["pipeline"]
        api.state.metadata = artifact["metadata"]
        yield

    api = FastAPI(title="Ticket Intelligence API", version="1.0.0", lifespan=lifespan)

    @api.middleware("http")
    async def request_timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
        # Do not log ticket bodies, which may contain personal information.
        logger.info("method=%s path=%s status=%s duration_ms=%.2f", request.method, request.url.path, response.status_code, elapsed)
        return response

    def classify(texts: list[str]) -> list[Prediction]:
        model = api.state.model
        probabilities = model.predict_proba(texts)
        results = []
        for row in probabilities:
            best = int(row.argmax())
            confidence = float(row[best])
            results.append(Prediction(
                category=str(model.classes_[best]),
                confidence=confidence,
                needs_review=confidence < review_threshold,
                probabilities={str(label): float(score) for label, score in zip(model.classes_, row)},
                model_version=api.state.metadata["model_version"],
            ))
        return results

    @api.get("/health")
    def health():
        return {"status": "ok", "model_version": api.state.metadata["model_version"]}

    @api.get("/model")
    def model_info():
        return {"metadata": api.state.metadata, "review_threshold": review_threshold}

    @api.post("/predict", response_model=Prediction)
    def predict(ticket: Ticket):
        return classify([ticket.text])[0]

    @api.post("/predict/batch", response_model=list[Prediction])
    def batch(payload: BatchRequest):
        return classify([ticket.text for ticket in payload.tickets])

    return api


app = create_app()

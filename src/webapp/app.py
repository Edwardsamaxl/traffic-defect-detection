from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

from src.webapp.api import auth, dashboard, detections, models  # users
from src.webapp.database import Base, SessionLocal, engine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "experiments/02_cbam/weights/best.pt"
UPLOADED_MODEL_DIR = ROOT / "experiments/uploaded_models"
OUTPUT_ROOT_DIR = ROOT / "output"
BATCH_OUTPUT_DIR = OUTPUT_ROOT_DIR / "webapp_batch_outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = ROOT / "data"

app = FastAPI(title="交通缺陷检测系统", version="2.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Root path redirect to index.html
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# Include routers
app.include_router(auth.router)
app.include_router(detections.router)
app.include_router(models.router)
app.include_router(dashboard.router)
# app.include_router(users.router)  # disabled - no admin/user roles

# In-memory model cache
_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_ERROR: str | None = None
_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_model_path(model: str | None) -> Path:
    if not model:
        return DEFAULT_MODEL_PATH.resolve()
    raw = Path(model)
    path = raw if raw.is_absolute() else ROOT / raw
    resolved = path.resolve()
    root_resolved = ROOT.resolve()
    if resolved.suffix.lower() != ".pt":
        raise ValueError("Only .pt model files are supported")
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise ValueError("Model path must be inside project root")
    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found: {resolved}")
    return resolved


def _load_model(model_path: Path | None = None) -> YOLO:
    global _MODEL_ERROR
    target_path = (model_path or DEFAULT_MODEL_PATH).resolve()
    cache_key = str(target_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if not target_path.exists():
        _MODEL_ERROR = f"Model file not found: {target_path}"
        raise FileNotFoundError(_MODEL_ERROR)
    model = YOLO(str(target_path))
    _MODEL_CACHE[cache_key] = model
    _MODEL_ERROR = None
    return model


def _safe_output_relative_path(raw_name: str) -> Path:
    normalized = raw_name.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    if not parts:
        return Path(f"image_{uuid4().hex[:8]}.jpg")
    rel_path = Path(*parts)
    suffix = rel_path.suffix.lower()
    if suffix not in _ALLOWED_IMAGE_EXTS:
        rel_path = rel_path.with_suffix(".jpg")
    return rel_path


def _run_prediction(image_bytes: bytes, conf: float, iou: float, imgsz: int, max_det: int, model_path: Path | None = None) -> dict[str, Any]:
    model = _load_model(model_path)
    image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    result = model.predict(source=image_array, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False)[0]
    boxes = result.boxes
    names = model.names
    detections_list = []
    if boxes is not None and len(boxes) > 0:
        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].item())
            score = float(boxes.conf[idx].item())
            xyxy = boxes.xyxy[idx].tolist()
            detections_list.append({
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(names[cls_id]),
                "confidence": round(score, 6),
                "bbox_xyxy": [round(float(v), 2) for v in xyxy],
            })
    plotted = result.plot()
    plotted_rgb = Image.fromarray(plotted[..., ::-1])
    import base64
    buffer = __import__("io").BytesIO()
    plotted_rgb.save(buffer, format="PNG")
    annotated_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "image_size": {"width": image.width, "height": image.height},
        "num_detections": len(detections_list),
        "detections": detections_list,
        "annotated_image_base64": annotated_b64,
    }


@app.on_event("startup")
def startup() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOADED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create DB tables
    Base.metadata.create_all(bind=engine)

    # Auto-register builtin model if not in DB
    from src.webapp.models.uploaded_model import UploadedModel
    with SessionLocal() as db:
        exists = db.query(UploadedModel).filter(
            UploadedModel.is_builtin == True,
            UploadedModel.path.like("%02_cbam%")
        ).first()
        if not exists:
            builtin = UploadedModel(
                name="02_cbam (内置)",
                path="experiments/02_cbam/weights/best.pt",
                is_builtin=True,
                uploaded_by=None,
            )
            db.add(builtin)
            db.commit()

    # Warm up default model
    try:
        _load_model()
    except Exception as exc:
        global _MODEL_ERROR
        _MODEL_ERROR = str(exc)


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    global _MODEL_ERROR
    healthy = str(DEFAULT_MODEL_PATH.resolve()) in _MODEL_CACHE and _MODEL_ERROR is None
    return {
        "status": "ok" if healthy else "degraded",
        "model_loaded": healthy,
        "model_path": str(DEFAULT_MODEL_PATH),
        "error": _MODEL_ERROR,
    }


# Legacy endpoints (kept for compatibility)
@app.get("/models")
def list_models_legacy() -> dict[str, Any]:
    models: list[str] = []
    for path in ROOT.rglob("*.pt"):
        try:
            rel = path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue
        if ".venv" in rel.parts or ".git" in rel.parts:
            continue
        models.append(str(rel).replace("\\", "/"))
    models.sort()
    default_rel = str(DEFAULT_MODEL_PATH.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    return {"default_model": default_rel, "models": models}


@app.post("/models/upload")
async def upload_model_legacy(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt model files are supported")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    UPLOADED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    original_name = Path(file.filename).stem
    safe_name = "".join(c for c in original_name if c.isalnum() or c in ("-", "_")) or "model"
    saved_name = f"{safe_name}-{uuid4().hex[:8]}.pt"
    save_path = UPLOADED_MODEL_DIR / saved_name
    save_path.write_bytes(content)
    rel = str(save_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    return JSONResponse({"message": "Model uploaded", "model_path": rel})


@app.post("/predict")
async def predict_legacy(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=0.99),
    iou: float = Query(default=0.6, ge=0.1, le=0.95),
    imgsz: int = Query(default=640, ge=320, le=1920),
    max_det: int = Query(default=300, ge=1, le=3000),
    model: str | None = Query(default=None),
) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        selected = _resolve_model_path(model)
        payload = _run_prediction(image_bytes, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, model_path=selected)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    payload["filename"] = file.filename or "uploaded_image"
    return JSONResponse(payload)


@app.post("/predict_batch")
async def predict_batch_legacy(
    files: list[UploadFile] = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=0.99),
    iou: float = Query(default=0.6, ge=0.1, le=0.95),
    imgsz: int = Query(default=640, ge=320, le=1920),
    max_det: int = Query(default=300, ge=1, le=3000),
    model: str | None = Query(default=None),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    try:
        selected = _resolve_model_path(model)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    run_dir = BATCH_OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    success_count = 0
    import base64
    from io import BytesIO
    for uploaded in files:
        filename = uploaded.filename or "unknown"
        if not uploaded.content_type or not uploaded.content_type.startswith("image/"):
            results.append({"filename": filename, "error": "Not an image"})
            continue
        image_bytes = await uploaded.read()
        if not image_bytes:
            results.append({"filename": filename, "error": "Empty file"})
            continue
        try:
            payload = _run_prediction(image_bytes, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, model_path=selected)
            annotated = Image.fromarray(Image.open(BytesIO(base64.b64decode(payload["annotated_image_base64"]))).convert("RGB"))
            rel_out = _safe_output_relative_path(filename)
            save_path = run_dir / rel_out
            save_path.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(save_path)
            results.append({
                "filename": filename,
                "num_detections": payload["num_detections"],
                "detections": payload["detections"],
                "saved_image": str(save_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
            })
            success_count += 1
        except Exception as exc:
            results.append({"filename": filename, "error": str(exc)})
    summary = {
        "model_path": str(selected.resolve()),
        "run_id": run_id,
        "output_dir": str(run_dir.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "total_files": len(files),
        "success_count": success_count,
        "failure_count": len(files) - success_count,
        "results": results,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return JSONResponse(summary)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.webapp.app:app", host="127.0.0.1", port=8000, reload=False)

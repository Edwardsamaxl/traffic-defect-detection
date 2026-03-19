from __future__ import annotations

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "experiments/stage4_overall/weights/best-cosine.pt"
UPLOADED_MODEL_DIR = ROOT / "experiments/uploaded_models"
OUTPUT_ROOT_DIR = ROOT / "output"
BATCH_OUTPUT_DIR = OUTPUT_ROOT_DIR / "webapp_batch_outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Traffic Defect Detection API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_ERROR: str | None = None
_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_model_path(model: str | None) -> Path:
    if not model:
        path = DEFAULT_MODEL_PATH
    else:
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


def _list_models() -> list[str]:
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
    return models


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


def _to_png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _run_prediction(
    image_bytes: bytes,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    model_path: Path | None = None,
    include_base64: bool = True,
    include_annotated_image: bool = False,
) -> dict[str, Any]:
    model = _load_model(model_path)

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid image file") from exc

    result = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )[0]

    boxes = result.boxes
    names = model.names
    detections: list[dict[str, Any]] = []

    if boxes is not None and len(boxes) > 0:
        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].item())
            score = float(boxes.conf[idx].item())
            xyxy = boxes.xyxy[idx].tolist()
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(names[cls_id]),
                    "confidence": round(score, 6),
                    "bbox_xyxy": [round(float(v), 2) for v in xyxy],
                }
            )

    plotted = result.plot()
    plotted_rgb = Image.fromarray(plotted[..., ::-1])

    payload: dict[str, Any] = {
        "image_size": {"width": image.width, "height": image.height},
        "num_detections": len(detections),
        "detections": detections,
        "model_path": str((model_path or DEFAULT_MODEL_PATH).resolve()),
    }
    if include_base64:
        payload["annotated_image_base64"] = _to_png_base64(plotted_rgb)
    if include_annotated_image:
        payload["annotated_image"] = plotted_rgb
    return payload


@app.on_event("startup")
def warmup() -> None:
    UPLOADED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _load_model()
    except Exception as exc:  # noqa: BLE001
        # Do not block service startup; health endpoint will expose details.
        global _MODEL_ERROR
        _MODEL_ERROR = str(exc)


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    global _MODEL_ERROR

    if str(DEFAULT_MODEL_PATH.resolve()) not in _MODEL_CACHE and _MODEL_ERROR is None:
        try:
            _load_model()
        except Exception as exc:  # noqa: BLE001
            _MODEL_ERROR = str(exc)

    healthy = str(DEFAULT_MODEL_PATH.resolve()) in _MODEL_CACHE and _MODEL_ERROR is None
    return {
        "status": "ok" if healthy else "degraded",
        "model_loaded": healthy,
        "model_path": str(DEFAULT_MODEL_PATH),
        "error": _MODEL_ERROR,
    }


@app.get("/models")
def list_models() -> dict[str, Any]:
    default_rel = str(DEFAULT_MODEL_PATH.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    return {
        "default_model": default_rel,
        "models": _list_models(),
    }


@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please choose a .pt file")
    if not file.filename.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt model files are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded model file is empty")

    UPLOADED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    original_name = Path(file.filename).stem
    safe_name = "".join(ch for ch in original_name if ch.isalnum() or ch in ("-", "_")) or "model"
    saved_name = f"{safe_name}-{uuid4().hex[:8]}.pt"
    save_path = UPLOADED_MODEL_DIR / saved_name
    save_path.write_bytes(content)

    relative = str(save_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    return JSONResponse({"message": "Model uploaded successfully", "model_path": relative})


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=0.99),
    iou: float = Query(default=0.6, ge=0.1, le=0.95),
    imgsz: int = Query(default=640, ge=320, le=1920),
    max_det: int = Query(default=300, ge=1, le=3000),
    model: str | None = Query(default=None, description="Relative model path under project root"),
) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        selected_model_path = _resolve_model_path(model)
        payload = _run_prediction(
            image_bytes,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            model_path=selected_model_path,
            include_base64=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    payload["filename"] = file.filename or "uploaded_image"
    return JSONResponse(payload)


@app.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=0.99),
    iou: float = Query(default=0.6, ge=0.1, le=0.95),
    imgsz: int = Query(default=640, ge=320, le=1920),
    max_det: int = Query(default=300, ge=1, le=3000),
    model: str | None = Query(default=None, description="Relative model path under project root"),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one image")

    try:
        selected_model_path = _resolve_model_path(model)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    run_dir = BATCH_OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    success_count = 0

    for uploaded in files:
        filename = uploaded.filename or "unknown"
        if not uploaded.content_type or not uploaded.content_type.startswith("image/"):
            results.append({"filename": filename, "error": "Not an image file"})
            continue

        image_bytes = await uploaded.read()
        if not image_bytes:
            results.append({"filename": filename, "error": "Uploaded file is empty"})
            continue

        try:
            payload = _run_prediction(
                image_bytes,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                model_path=selected_model_path,
                include_base64=False,
                include_annotated_image=True,
            )
            annotated_image = payload.pop("annotated_image")
            rel_output_path = _safe_output_relative_path(filename)
            save_path = run_dir / rel_output_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_image.save(save_path)

            payload["filename"] = filename
            payload["saved_image"] = str(save_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
            results.append(payload)
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            results.append({"filename": filename, "error": str(exc)})

    summary = {
        "model_path": str(selected_model_path.resolve()),
        "run_id": run_id,
        "output_dir": str(run_dir.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "total_files": len(files),
        "success_count": success_count,
        "failure_count": len(files) - success_count,
        "results": results,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return JSONResponse(
        summary
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.webapp.app:app", host="127.0.0.1", port=8000, reload=False)

from __future__ import annotations

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ultralytics import YOLO

from src.webapp.database import get_db
from src.webapp.middleware.auth import get_current_user
from src.webapp.models.detection_record import DetectionRecord
from src.webapp.models.user import User
from src.webapp.models.uploaded_model import UploadedModel

router = APIRouter(prefix="/api/detections", tags=["detections"])

ROOT = Path(__file__).resolve().parents[3]
BATCH_OUTPUT_DIR = ROOT / "output" / "webapp_batch_outputs"
BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE: dict[str, YOLO] = {}


def _load_model(model_path: str | None = None) -> YOLO:
    global _MODEL_CACHE
    if model_path:
        cache_key = model_path
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        if not resolved.exists():
            raise FileNotFoundError(f"Model not found: {resolved}")
        model = YOLO(str(resolved))
        _MODEL_CACHE[cache_key] = model
        return model

    default_path = ROOT / "experiments/02_cbam/weights/best.pt"
    cache_key = str(default_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    model = YOLO(str(default_path))
    _MODEL_CACHE[cache_key] = model
    return model


def _safe_output_relative_path(raw_name: str) -> Path:
    normalized = raw_name.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    if not parts:
        return Path(f"image_{uuid4().hex[:8]}.jpg")
    rel_path = Path(*parts)
    suffix = rel_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        rel_path = rel_path.with_suffix(".jpg")
    return rel_path


def _run_prediction(image_bytes: bytes, conf: float, iou: float, imgsz: int, max_det: int, model_path: str | None = None) -> dict[str, Any]:
    model = _load_model(model_path)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)

    result = model.predict(source=image_array, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False)[0]
    boxes = result.boxes
    names = model.names
    detections = []

    if boxes is not None and len(boxes) > 0:
        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].item())
            score = float(boxes.conf[idx].item())
            xyxy = boxes.xyxy[idx].tolist()
            detections.append({
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(names[cls_id]),
                "confidence": round(score, 6),
                "bbox_xyxy": [round(float(v), 2) for v in xyxy],
            })

    plotted = result.plot()
    plotted_rgb = Image.fromarray(plotted[..., ::-1])
    buffer = BytesIO()
    plotted_rgb.save(buffer, format="PNG")
    annotated_png_bytes = buffer.getvalue()

    return {
        "image_size": {"width": image.width, "height": image.height},
        "num_detections": len(detections),
        "detections": detections,
        "annotated_image_base64": base64.b64encode(annotated_png_bytes).decode("utf-8"),
        "_annotated_png_bytes": annotated_png_bytes,  # internal use, not serialized to JSON
    }


class PredictResponse(BaseModel):
    record_id: int
    detections: list
    num_detections: int
    image_size: dict
    annotated_image_base64: str
    model_name: str


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=0.99),
    iou: float = Query(default=0.6, ge=0.1, le=0.95),
    imgsz: int = Query(default=640, ge=320, le=1920),
    max_det: int = Query(default=300, ge=1, le=3000),
    model_id: int | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="图片为空")

    model_name = "默认模型"
    resolved_model_path = None

    if model_id is not None:
        db_model = db.query(UploadedModel).filter(UploadedModel.id == model_id).first()
        if db_model:
            model_name = db_model.name
            resolved_model_path = str(ROOT / db_model.path)

    try:
        result_data = _run_prediction(image_bytes, conf, iou, imgsz, max_det, resolved_model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # 检测失败时返回原图 + 0 检测数，前端按正常流程处理
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            annotated_png_bytes = buffer.getvalue()
            result_data = {
                "image_size": {"width": image.width, "height": image.height},
                "num_detections": 0,
                "detections": [],
                "annotated_image_base64": base64.b64encode(annotated_png_bytes).decode("utf-8"),
                "_annotated_png_bytes": annotated_png_bytes,
            }
        except Exception:
            # Fallback 也失败时返回最小有效响应，避免 500
            result_data = {
                "image_size": {"width": 0, "height": 0},
                "num_detections": 0,
                "detections": [],
                "annotated_image_base64": "",
                "_annotated_png_bytes": b"",
            }

    try:
        record = DetectionRecord(
            user_id=current_user.id,
            filename=file.filename or "uploaded_image",
            model_name=model_name,
            conf=conf,
            iou=iou,
            num_detections=result_data["num_detections"],
            detections=json.dumps(result_data["detections"], ensure_ascii=False),
            image_width=result_data["image_size"]["width"],
            image_height=result_data["image_size"]["height"],
            annotated_image_base64=result_data["annotated_image_base64"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        record_id = record.id
    except Exception:
        db.rollback()
        record_id = 0

    return PredictResponse(
        record_id=record_id,
        detections=result_data["detections"],
        num_detections=result_data["num_detections"],
        image_size=result_data["image_size"],
        annotated_image_base64=result_data["annotated_image_base64"],
        model_name=model_name,
    )


@router.post("/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    conf: float = Query(default=0.25, ge=0.01, le=0.99),
    iou: float = Query(default=0.6, ge=0.1, le=0.95),
    imgsz: int = Query(default=640, ge=320, le=1920),
    max_det: int = Query(default=300, ge=1, le=3000),
    model_id: int | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一张图片")

    model_name = "默认模型"
    resolved_model_path = None

    if model_id is not None:
        db_model = db.query(UploadedModel).filter(UploadedModel.id == model_id).first()
        if db_model:
            model_name = db_model.name
            resolved_model_path = str(ROOT / db_model.path)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    run_dir = BATCH_OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    success_count = 0

    for uploaded in files:
        filename = uploaded.filename or "unknown"
        if not uploaded.content_type or not uploaded.content_type.startswith("image/"):
            results.append({"filename": filename, "error": "Not an image"})
            continue

        image_bytes = await uploaded.read()
        # Defensive: ensure bytes (guard against string/Image leakage from malformed multipart)
        if not isinstance(image_bytes, bytes):
            image_bytes = str(image_bytes).encode("latin-1")
        if not image_bytes:
            results.append({"filename": filename, "error": "Empty file"})
            continue

        try:
            payload = _run_prediction(image_bytes, conf, iou, imgsz, max_det, resolved_model_path)
        except Exception as exc:
            # 检测失败时返回原图 + 0 检测数
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            annotated_png_bytes = buffer.getvalue()
            payload = {
                "image_size": {"width": image.width, "height": image.height},
                "num_detections": 0,
                "detections": [],
                "_annotated_png_bytes": annotated_png_bytes,
            }

        # Use raw PNG bytes directly (avoids broken base64→Image→fromarray chain)
        annotated_png_bytes = payload.get("_annotated_png_bytes") or base64.b64decode(payload["annotated_image_base64"])
        rel_output_path = _safe_output_relative_path(filename)
        save_path = run_dir / rel_output_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(annotated_png_bytes)

        # 批量模式图片已存磁盘，base64 存空节省数据库空间
        record = DetectionRecord(
            user_id=current_user.id,
            filename=filename,
            model_name=model_name,
            conf=conf,
            iou=iou,
            num_detections=payload["num_detections"],
            detections=json.dumps(payload["detections"], ensure_ascii=False),
            image_width=payload["image_size"]["width"],
            image_height=payload["image_size"]["height"],
            annotated_image_base64="",
        )
        db.add(record)
        db.commit()

        results.append({
            "filename": filename,
            "num_detections": payload["num_detections"],
            "detections": payload["detections"],
            "saved_image": str(save_path.relative_to(ROOT)).replace("\\", "/"),
        })
        success_count += 1

    summary = {
        "run_id": run_id,
        "output_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "total_files": len(files),
        "success_count": success_count,
        "failure_count": len(files) - success_count,
        "results": results,
    }
    return summary


class RecordListResponse(BaseModel):
    records: list
    total: int
    page: int
    limit: int


@router.get("")
def list_detections(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    search: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RecordListResponse:
    query = db.query(DetectionRecord).filter(DetectionRecord.user_id == current_user.id)

    if search:
        query = query.filter(DetectionRecord.filename.contains(search))
    if model_name:
        query = query.filter(DetectionRecord.model_name == model_name)
    if date_from:
        query = query.filter(DetectionRecord.created_at >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(DetectionRecord.created_at <= datetime.fromisoformat(date_to))

    total = query.count()
    records = (
        query
        .order_by(DetectionRecord.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    return RecordListResponse(
        records=[r.to_dict() for r in records],
        total=total,
        page=page,
        limit=limit,
    )


@router.delete("/clear")
def clear_detections(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    db.query(DetectionRecord).filter(DetectionRecord.user_id == current_user.id).delete()
    db.commit()
    return {"message": "已清除所有检测记录"}


@router.get("/{record_id}")
def get_detection(
    record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    record = db.query(DetectionRecord).filter(
        DetectionRecord.id == record_id,
        DetectionRecord.user_id == current_user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    return record.to_dict(include_image=True)

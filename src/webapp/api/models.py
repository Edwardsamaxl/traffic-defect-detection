from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.webapp.database import get_db
from src.webapp.middleware.auth import get_current_admin, get_current_user
from src.webapp.models.uploaded_model import UploadedModel
from src.webapp.models.user import User

router = APIRouter(prefix="/api/models", tags=["models"])

ROOT = Path(__file__).resolve().parents[3]
UPLOADED_MODEL_DIR = ROOT / "experiments/uploaded_models"
UPLOADED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


@router.get("")
def list_models(
    _: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    # List models from uploaded_models table + project .pt files
    db_models = db.query(UploadedModel).order_by(UploadedModel.uploaded_at.desc()).all()

    # Build uploader_id -> username map
    uploader_ids = {m.uploaded_by for m in db_models if m.uploaded_by}
    users_map = {u.id: u.username for u in db.query(User).filter(User.id.in_(uploader_ids)).all()} if uploader_ids else {}

    # Scan project for .pt files not yet in DB
    project_models = []
    for pt_path in ROOT.rglob("*.pt"):
        try:
            rel = pt_path.relative_to(ROOT)
        except ValueError:
            continue
        if ".venv" in rel.parts or ".git" in rel.parts or "uploaded_models" in rel.parts:
            continue
        path_str = str(rel).replace("\\", "/")
        if not db.query(UploadedModel).filter(UploadedModel.path == path_str).first():
            project_models.append({
                "id": f"proj_{path_str}",
                "name": pt_path.name,
                "path": path_str,
                "uploaded_by": None,
                "uploaded_by_name": None,
                "uploaded_at": None,
            })

    return {
        "models": [m.to_dict(uploader_name=users_map.get(m.uploaded_by)) for m in db_models] + project_models,
    }


@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="仅支持 .pt 文件")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")

    safe_name = "".join(c for c in file.filename if c.isalnum() or c in ("-", "_", ".")) or "model"
    saved_name = f"{safe_name[:-3]}_{Path(file.filename).stem[:20]}.pt"
    save_path = UPLOADED_MODEL_DIR / saved_name
    save_path.write_bytes(content)

    rel_path = str(save_path.relative_to(ROOT)).replace("\\", "/")

    db_model = UploadedModel(
        name=file.filename,
        path=rel_path,
        uploaded_by=current_user.id,
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return {"model": db_model.to_dict()}


@router.delete("/{model_id}")
def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if model_id.startswith("proj_"):
        raise HTTPException(status_code=400, detail="无法删除项目内置模型")

    model = db.query(UploadedModel).filter(UploadedModel.id == int(model_id)).first()
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    # Only admin or the uploader can delete
    if current_user.role != "admin" and model.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="无权删除此模型")

    # Delete file
    file_path = ROOT / model.path
    if file_path.exists():
        file_path.unlink()

    db.delete(model)
    db.commit()
    return {"message": "模型已删除"}

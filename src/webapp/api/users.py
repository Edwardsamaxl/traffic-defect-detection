from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.webapp.database import get_db
from src.webapp.middleware.auth import get_current_admin, get_current_user
from src.webapp.models.user import User

router = APIRouter(prefix="/api/users", tags=["users"])


class UpdateRoleRequest(BaseModel):
    role: str


@router.get("")
def list_users(
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> dict:
    users = db.query(User).order_by(User.created_at.desc()).all()
    return {"users": [u.to_dict() for u in users]}


@router.put("/{user_id}/role")
def update_role(
    user_id: int,
    body: UpdateRoleRequest,
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> dict:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if body.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="无效的角色")
    user.role = body.role
    db.commit()
    return {"user": user.to_dict()}


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> dict:
    if current_admin.id == user_id:
        raise HTTPException(status_code=400, detail="不能删除自己")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    db.delete(user)
    db.commit()
    return {"message": "用户已删除"}

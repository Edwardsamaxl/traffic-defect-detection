from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from src.webapp.database import Base


class UploadedModel(Base):
    __tablename__ = "uploaded_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    path = Column(String(500), nullable=False, unique=True)
    is_builtin = Column(Boolean, nullable=False, default=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    uploader = relationship("User", back_populates="uploaded_models")

    def to_dict(self, uploader_name: str | None = None) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "is_builtin": self.is_builtin,
            "uploaded_by": self.uploaded_by,
            "uploaded_by_name": uploader_name,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
        }

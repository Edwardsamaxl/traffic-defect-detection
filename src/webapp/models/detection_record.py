from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from src.webapp.database import Base


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    conf = Column(Float, nullable=False)
    iou = Column(Float, nullable=False)
    num_detections = Column(Integer, nullable=False)
    detections = Column(JSON, nullable=False)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    annotated_image_base64 = Column(Text, nullable=True)  # only for single-image view
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="detection_records")

    def to_dict(self, include_image: bool = False) -> dict:
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "filename": self.filename,
            "model_name": self.model_name,
            "conf": self.conf,
            "iou": self.iou,
            "num_detections": self.num_detections,
            "detections": json.loads(self.detections) if self.detections else [],
            "image_size": {"width": self.image_width, "height": self.image_height},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_image and self.annotated_image_base64:
            data["annotated_image_base64"] = self.annotated_image_base64
        return data

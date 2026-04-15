from __future__ import annotations

from src.webapp.database import Base
from src.webapp.models.detection_record import DetectionRecord
from src.webapp.models.uploaded_model import UploadedModel
from src.webapp.models.user import User

__all__ = ["Base", "User", "DetectionRecord", "UploadedModel"]

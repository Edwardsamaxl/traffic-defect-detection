from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from src.webapp.database import get_db
from src.webapp.middleware.auth import get_current_user
from src.webapp.models.detection_record import DetectionRecord
from src.webapp.models.user import User

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats")
def get_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    today = datetime.utcnow().date()
    seven_days_ago = datetime.utcnow() - timedelta(days=6)

    # Base query (always filter by user - no admin role exists)
    base_filter = DetectionRecord.user_id == current_user.id

    # Total detections
    total = db.query(func.count(DetectionRecord.id)).filter(base_filter).scalar() or 0

    # Today's detections
    today_count = (
        db.query(func.count(DetectionRecord.id))
        .filter(base_filter, func.date(DetectionRecord.created_at) == today)
        .scalar()
        or 0
    )

    # Last 30 days: detections per day (fill missing days with 0)
    daily_data = (
        db.query(
            func.date(DetectionRecord.created_at).label("day"),
            func.count(DetectionRecord.id).label("count"),
        )
        .filter(base_filter, DetectionRecord.created_at >= seven_days_ago)
        .group_by(text("day"))
        .order_by(text("day"))
        .all()
    )
    daily_map = {str(row.day): row.count for row in daily_data}
    by_day = [
        {"date": str((seven_days_ago + timedelta(days=i)).date()), "count": daily_map.get(str((seven_days_ago + timedelta(days=i)).date()), 0)}
        for i in range(7)
    ]

    # Last 30 days: class distribution (for bar chart)
    records = (
        db.query(DetectionRecord.detections)
        .filter(base_filter, DetectionRecord.created_at >= seven_days_ago)
        .all()
    )

    class_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record.detections:
            try:
                detections = json.loads(record.detections) if isinstance(record.detections, str) else record.detections
                for det in detections:
                    class_name = det.get("class_name", "unknown")
                    class_counts[class_name] += 1
            except (json.JSONDecodeError, TypeError):
                pass

    by_class = sorted(
        [{"class_name": k, "count": v} for k, v in class_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:10]

    return {
        "total_detections": total,
        "detections_today": today_count,
        "by_day": by_day,
        "by_class": by_class,
    }

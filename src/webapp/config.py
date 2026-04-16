from __future__ import annotations

import os

# Fixed secret for development - in production use environment variable
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-for-traffic-defect-detection-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours


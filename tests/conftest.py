from __future__ import annotations

import os

# Configure settings BEFORE importing the app.
os.environ.setdefault("TS_ENVIRONMENT", "test")
os.environ.setdefault("TS_SECRET_KEY", "test-secret-key-that-is-long-enough-1234567890")
os.environ.setdefault("TS_CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("TS_LOG_LEVEL", "WARNING")

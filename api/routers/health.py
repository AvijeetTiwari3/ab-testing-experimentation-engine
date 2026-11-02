"""Health monitoring endpoint"""

from fastapi import APIRouter
import platform
import sys

router = APIRouter()


@router.get("/health", tags=["Monitoring"])
def health_check():
    return {
        "status": "HEALTHY",
        "service": "ab-testing-experimentation-engine",
        "version": "1.0.0",
        "python_version": sys.version,
        "platform": platform.platform()
    }

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, REGISTRY

router = APIRouter()

@router.get("/metrics")
async def metrics():
    """Endpoint to expose metrics for Prometheus"""
    return Response(
        generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )



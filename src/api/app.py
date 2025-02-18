from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from src.config import get_settings
from src.config.constants import API_DESCRIPTION


settings = get_settings()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=API_DESCRIPTION,
        version="1.0.0",
        openapi_url=f"/{settings.API_V1_STR}/openapi.json",
        docs_url=f"/{settings.API_V1_STR}/docs",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Add exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
            },
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy"}

    # Import and include API routes
    from .routes import prediction, metrics_endpoint
    app.include_router(
        prediction.router,
        prefix=settings.API_V1_STR,
        tags=["predictions"]
    )
    app.include_router(
        metrics_endpoint.router,
    )

    return app

# Create application instance
app = create_app()
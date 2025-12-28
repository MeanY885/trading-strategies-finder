"""
API ROUTES PACKAGE
==================
Modular API route definitions for the BTCGBP ML Optimizer.
"""
from fastapi import APIRouter

from .data_routes import router as data_router
from .system_routes import router as system_router
from .optimization_routes import router as optimization_router
from .autonomous_routes import router as autonomous_router
from .elite_routes import router as elite_router
from .priority_routes import router as priority_router
from .db_routes import router as db_router
from .comparison_routes import router as comparison_router
from .validation_routes import router as validation_router


def register_routes(app):
    """Register all API routes with the FastAPI app."""
    app.include_router(data_router)
    app.include_router(system_router)
    app.include_router(optimization_router)
    app.include_router(autonomous_router)
    app.include_router(elite_router)
    app.include_router(priority_router)
    app.include_router(db_router)
    app.include_router(comparison_router)
    app.include_router(validation_router)


__all__ = [
    'data_router',
    'system_router',
    'optimization_router',
    'autonomous_router',
    'elite_router',
    'priority_router',
    'db_router',
    'comparison_router',
    'validation_router',
    'register_routes',
]

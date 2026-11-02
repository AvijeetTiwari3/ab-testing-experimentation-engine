"""
FastAPI Main Application for A/B Testing & Statistical Experimentation Analytics Engine
"""

import sys
import os
from contextlib import asynccontextmanager

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import health, power, analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("A/B Testing & Statistical Experimentation Engine API Online")
    print("Google-standard dual Frequentist & Bayesian architecture")
    print("=" * 60)
    yield
    print("A/B Testing Engine API shutting down cleanly.")


app = FastAPI(
    title="A/B Testing & Statistical Experimentation Analytics Engine",
    description=(
        "Production-grade statistical experimentation platform with dual Frequentist "
        "(z-test, chi-square, Mann-Whitney U) & Bayesian (Beta-Binomial conjugate, HDI, Expected Loss) "
        "engines, O'Brien-Fleming sequential testing, and FDR multi-testing corrections."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for web dashboards and cross-origin tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(power.router, prefix="/power-analysis", tags=["Power Analysis & Sample Sizing"])
app.include_router(analysis.router, prefix="/analyze", tags=["Statistical Experimentation"])


@app.get("/", tags=["Root"])
def root_endpoint():
    return {
        "engine": "A/B Testing & Statistical Experimentation Analytics Engine",
        "author": "Avijeet Tiwari (Google Data Scientist Portfolio Standard)",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "empirical_readout": "/analyze/empirical-readout"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

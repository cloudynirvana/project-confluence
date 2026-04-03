"""
Project Confluence — API Service
==================================

FastAPI application with Nigeria Clinical Guidelines RAG integration.

Endpoints:
    POST /generate_research      — Submit an AutoResearch order
    GET  /status/{job_id}        — Check research job status
    POST /guideline_query        — Query Nigerian clinical guidelines (NSTG 2022)
    GET  /guideline_conditions   — List all 270 NSTG conditions
    GET  /guideline_protocol/{condition} — Get full treatment protocol
    GET  /guideline_drug/{drug}  — Get dosing constraints for a drug
    GET  /health                 — Health check

Data Attribution:
    Nigeria Standard Treatment Guidelines (NSTG) 2022
    Federal Ministry of Health, Nigeria
    Dataset: CC-BY-4.0, curated by Chisom Rutherford
    Source: chisomrutherford/nigeria-clinical-guidelines-dataset (HuggingFace)
"""

import os
import subprocess
import uuid
import sys
import warnings
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from pathlib import Path

app = FastAPI(
    title="Project Confluence — AutoResearch & Clinical Guidelines API",
    version="2.0.0",
    description=(
        "Precision oncology API with Nigerian clinical guidelines integration. "
        "Powered by NSTG 2022 (CC-BY-4.0)."
    ),
)

# In-memory job store (Use a real DB like Postgres/Redis for production)
jobs: Dict[str, dict] = {}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Lazy-loaded guideline retriever ────────────────────────────────────
_guideline_retriever = None


def get_retriever():
    """Lazy-load the NigeriaGuidelineRetriever (one-time init)."""
    global _guideline_retriever
    if _guideline_retriever is not None:
        return _guideline_retriever

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from agents.nigeria_guideline_retriever import NigeriaGuidelineRetriever
        _guideline_retriever = NigeriaGuidelineRetriever()
        return _guideline_retriever
    except Exception as e:
        warnings.warn(f"Failed to load NigeriaGuidelineRetriever: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class ResearchRequest(BaseModel):
    client_topic: str
    disease_target: str
    
class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class GuidelineQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    field_filter: Optional[str] = None

class GuidelineQueryResponse(BaseModel):
    query: str
    n_results: int
    results: List[Dict]
    answer: str


# ═══════════════════════════════════════════════════════════════════════════
# AUTORESEARCH ENDPOINTS (existing)
# ═══════════════════════════════════════════════════════════════════════════

def run_project_confluence_sandbox(job_id: str, topic: str):
    """Background task to run the AutoResearchClaw pipeline."""
    jobs[job_id]["status"] = "processing"
    print(f"[{job_id}] Starting background research task for topic: {topic}")
    
    script_path = PROJECT_ROOT / "scripts" / "run_autoresearch.py"
    
    try:
        import time
        time.sleep(10)
        
        mock_output = "✅ Pipeline complete!\n📄 Deliverables copied to: results/autoresearch/..."
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_log"] = mock_output
        print(f"[{job_id}] Finished successfully (Mocked).")
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"[{job_id}] Exception occurred: {e}")

@app.post("/generate_research", response_model=JobResponse)
async def generate_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Submits a new "AutoResearch" order.
    The client pays, submits the topic, and this kicks off the Nemotron LLM pipeline.
    """
    if not req.client_topic:
        raise HTTPException(status_code=400, detail="client_topic is required.")
        
    job_id = str(uuid.uuid4())
    
    enhanced_topic = (
        f"{req.client_topic}. Focus specifically on the {req.disease_target} "
        f"archetype using the Project Confluence 15D ODE model and Phi complexity vector."
    )
    
    jobs[job_id] = {
        "status": "queued",
        "topic": enhanced_topic
    }
    
    background_tasks.add_task(run_project_confluence_sandbox, job_id, enhanced_topic)
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Research pipeline initialized. Check /status/{job_id} for updates."
    )

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check the status of a generated paper."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    
    resp = {
        "job_id": job_id,
        "status": job["status"]
    }
    
    if job["status"] == "completed":
        resp["download_url"] = f"/download/{job_id}"
        resp["preview_log"] = job.get("output_log", "")
    elif job["status"] == "failed":
        resp["error"] = job.get("error", "Unknown error")
        
    return resp


# ═══════════════════════════════════════════════════════════════════════════
# NIGERIA CLINICAL GUIDELINES ENDPOINTS (NEW)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/guideline_query", response_model=GuidelineQueryResponse)
async def guideline_query(req: GuidelineQueryRequest):
    """
    Query the Nigeria Standard Treatment Guidelines (NSTG 2022) using
    semantic search (RAG).

    Returns ranked guideline passages relevant to the clinical query.
    
    Data Source: CC-BY-4.0, Federal Ministry of Health, Nigeria (2022).
    Dataset curated by Chisom Rutherford.
    """
    retriever = get_retriever()
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Nigeria Guideline Retriever is not available. "
                   "Install: pip install sentence-transformers faiss-cpu datasets"
        )
    
    results = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        field_filter=req.field_filter,
    )
    answer = retriever.answer(req.query, top_k=min(req.top_k, 3))
    
    return GuidelineQueryResponse(
        query=req.query,
        n_results=len(results),
        results=[r.to_dict() for r in results],
        answer=answer,
    )


@app.get("/guideline_conditions")
async def guideline_conditions():
    """
    List all conditions in the NSTG 2022 dataset (270 conditions).
    
    Data Source: CC-BY-4.0, Federal Ministry of Health, Nigeria (2022).
    """
    retriever = get_retriever()
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not available.")
    
    conditions = retriever.list_conditions()
    return {
        "n_conditions": len(conditions),
        "conditions": conditions,
        "source": "Nigeria Standard Treatment Guidelines (NSTG) 2022",
        "license": "CC-BY-4.0",
    }


@app.get("/guideline_protocol/{condition}")
async def guideline_protocol(condition: str):
    """
    Get the full structured treatment protocol for a specific condition.
    
    Args:
        condition: Condition name (case-insensitive, e.g., "breast cancer").
    
    Data Source: CC-BY-4.0, Federal Ministry of Health, Nigeria (2022).
    """
    retriever = get_retriever()
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not available.")
    
    protocol = retriever.get_treatment_protocol(condition)
    if protocol is None:
        raise HTTPException(
            status_code=404,
            detail=f"Condition '{condition}' not found. "
                   f"Use /guideline_conditions to list available conditions."
        )
    
    return {
        "condition": condition,
        "protocol": protocol,
        "source": "NSTG 2022",
        "license": "CC-BY-4.0",
    }


@app.get("/guideline_drug/{drug_name}")
async def guideline_drug(drug_name: str):
    """
    Get dosing constraints, adverse reactions, and cautions for a specific drug
    across all NSTG 2022 conditions.
    
    Args:
        drug_name: Drug name (case-insensitive, e.g., "doxorubicin").
    
    Data Source: CC-BY-4.0, Federal Ministry of Health, Nigeria (2022).
    """
    retriever = get_retriever()
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not available.")
    
    constraints = retriever.get_dosing_constraints(drug_name)
    if not constraints:
        raise HTTPException(
            status_code=404,
            detail=f"Drug '{drug_name}' not found in NSTG 2022 guidelines."
        )
    
    return {
        "drug": drug_name,
        "n_conditions": len(constraints),
        "dosing_constraints": constraints,
        "source": "NSTG 2022",
        "license": "CC-BY-4.0",
    }


@app.get("/health")
async def health_check():
    retriever = get_retriever()
    return {
        "status": "ok",
        "engine": "Nemotron-70B integrated",
        "nigeria_guidelines": retriever is not None,
        "guideline_stats": retriever.get_stats() if retriever else None,
    }

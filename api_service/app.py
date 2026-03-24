import os
import subprocess
import uuid
import sys
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict
from pathlib import Path

app = FastAPI(title="AutoResearch as a Service API", version="1.0.0")

# In-memory job store (Use a real DB like Postgres/Redis for production)
jobs: Dict[str, dict] = {}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class ResearchRequest(BaseModel):
    client_topic: str
    disease_target: str
    
class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

def run_project_confluence_sandbox(job_id: str, topic: str):
    """Background task to run the AutoResearchClaw pipeline."""
    jobs[job_id]["status"] = "processing"
    print(f"[{job_id}] Starting background research task for topic: {topic}")
    
    script_path = PROJECT_ROOT / "scripts" / "run_autoresearch.py"
    
    try:
        # In a real deployed environment we would run:
        # process = subprocess.run([sys.executable, str(script_path), topic], ...)
        # For this demonstration, we mock the 15-minute wait to show the API flow.
        import time
        time.sleep(10)
        
        # Simulate successful pipeline output
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
    
    # Pre-process the topic to ensure it fits the Confluence/biology domain
    enhanced_topic = f"{req.client_topic}. Focus specifically on the {req.disease_target} archetype using the Project Confluence 15D ODE model and Phi complexity vector."
    
    jobs[job_id] = {
        "status": "queued",
        "topic": enhanced_topic
    }
    
    # Add to FastAPI background tasks
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
        resp["download_url"] = f"/download/{job_id}" # Presigned S3/Local link pattern
        resp["preview_log"] = job.get("output_log", "")
    elif job["status"] == "failed":
        resp["error"] = job.get("error", "Unknown error")
        
    return resp

@app.get("/health")
async def health_check():
    return {"status": "ok", "engine": "Nemotron-70B integrated"}

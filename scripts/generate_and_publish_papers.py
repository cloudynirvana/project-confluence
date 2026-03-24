import os
import glob
import json
import subprocess
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Paths to projects
PROJECT_CONFLUENCE_DIR = Path(__file__).parent.parent
AUTORESEARCHCLAW_DIR = PROJECT_CONFLUENCE_DIR.parent / "AutoResearchClaw"
CALIBRATION_RESULTS_DIR = PROJECT_CONFLUENCE_DIR / "results" / "calibration"

def get_latest_calibration_result():
    """Finds the most recent JSON file in the calibration results directory."""
    if not CALIBRATION_RESULTS_DIR.exists():
        raise FileNotFoundError(f"Calibration directory not found: {CALIBRATION_RESULTS_DIR}")
        
    json_files = list(CALIBRATION_RESULTS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No calibration JSON files found in {CALIBRATION_RESULTS_DIR}")
        
    latest_file = max(json_files, key=os.path.getctime)
    return latest_file

def parse_calibration_data(filepath):
    """Extracts leading disease and targets from the calibration JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    diseases = data.get("per_disease_results", {})
    if not diseases:
        raise ValueError("Invalid calibration JSON format: 'per_disease_results' key missing.")
        
    # Sort diseases by calibration score (lower is better, assuming Phi distance difference)
    # Actually let's use the one with the highest delta_phi or simply pick the first one with the best stability
    sorted_diseases = sorted(
        diseases.items(),
        key=lambda item: sum(item[1].get("stability", {}).values()) / len(item[1].get("stability", {}) or [1]),
        reverse=True
    )
    
    top_disease_name, top_disease_data = sorted_diseases[0]
    
    # Extract top targets based on druggability from sis_vectors
    sis_vectors = top_disease_data.get("sis_vectors", {})
    
    # Sort genes by druggability score * plddt
    sorted_genes = sorted(
        sis_vectors.values(),
        key=lambda g: g.get("druggability", 0) * g.get("plddt", 0),
        reverse=True
    )
    
    top_targets = [g["gene"] for g in sorted_genes[:3]]
    
    return top_disease_name, top_targets

def construct_research_topic(disease, targets):
    """Creates a high-quality academic research topic based on the data."""
    targets_str = ", ".join(targets)
    topic = (
        f"Structural Druggability and Predictive Neural ODE Modeling of "
        f"{targets_str} in the context of {disease}. Focus on mapping AlphaFold "
        f"confidence scores (pLDDT) and pocket geometry directly to dynamical systems "
        f"parameters for intervention simulation."
    )
    return topic

def run_autoresearchclaw(topic):
    """Executes the AutoResearchClaw pipeline via CLI."""
    print(f"\n[+] Executing AutoResearchClaw with topic:\n'{topic}'")
    
    # We must run it from the AutoResearchClaw directory
    current_dir = os.getcwd()
    try:
        os.chdir(AUTORESEARCHCLAW_DIR)
        
        # Command array
        cmd = [
            # Assumes the local python environment is active and 'researchclaw' is globally available,
            # or we call python -m researchclaw.pipeline if it isn't installed.
            # Using python -m for safety:
            "python", "-m", "researchclaw.cli", "run",
            "--topic", topic,
            "--auto-approve"
        ]
        
        # We use Popen to stream the output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[!] AutoResearchClaw failed with exit code {process.returncode}")
            return False
            
        return True
    finally:
        os.chdir(current_dir)

def find_deliverables_folder():
    """Locates the newly generated deliverables folder in AutoResearchClaw artifacts."""
    artifacts_dir = AUTORESEARCHCLAW_DIR / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
        
    run_folders = [f for f in artifacts_dir.iterdir() if f.is_dir() and f.name.startswith("rc-")]
    if not run_folders:
        raise FileNotFoundError("No run folders found in artifacts directory.")
        
    latest_run_folder = max(run_folders, key=lambda f: f.stat().st_ctime)
    deliverables_dir = latest_run_folder / "deliverables"
    
    if not deliverables_dir.exists():
        raise FileNotFoundError(f"Deliverables directory not found in: {latest_run_folder}")
        
    return deliverables_dir

def publish_to_github(deliverables_dir, repo_url):
    """Initializes a git repo in the deliverables dir and forces a push to a remote repo."""
    print(f"\n[+] Publishing {deliverables_dir} to GitHub: {repo_url}")
    
    current_dir = os.getcwd()
    try:
        os.chdir(deliverables_dir)
        
        # Initialize git
        subprocess.run(["git", "init"], check=True, capture_output=True)
        
        # Remote config - replace origin if exists
        subprocess.run(["git", "remote", "remove", "origin"], capture_output=True) # Ignore errors if it doesn't exist
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True, capture_output=True)
        
        # Add and commit
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        
        commit_msg = f"AutoResearchClaw Publication: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Check if there is anything to commit
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not status.stdout.strip():
            print("[+] No new changes to commit.")
            return True
            
        subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
        
        # Push to main forcefully (since this is an automated publications repo)
        # Assuming the default branch is main
        print("[+] Pushing to remote repository...")
        push_result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], capture_output=True, text=True)
        
        if push_result.returncode != 0:
             print(f"[!] Push failed: {push_result.stderr}")
             # try master if main fails
             push_master = subprocess.run(["git", "push", "-u", "origin", "master", "--force"], capture_output=True, text=True)
             if push_master.returncode != 0:
                 print(f"[!] Push to master also failed: {push_master.stderr}")
                 return False
                 
        print("[+] Successfully published to GitHub!")
        return True
        
    finally:
        os.chdir(current_dir)

def main():
    parser = argparse.ArgumentParser(description="Bridge Project Confluence with AutoResearchClaw")
    parser.add_argument("--repo-url", type=str, required=True, help="GitHub repository URL to publish the paper to (e.g. https://github.com/user/papers.git)")
    parser.add_argument("--dry-run", action="store_true", help="Format the topic but do not run the LLM pipeline")
    args = parser.parse_args()
    
    try:
        latest_file = get_latest_calibration_result()
        print(f"[+] Found latest calibration data: {latest_file.name}")
        
        disease, targets = parse_calibration_data(latest_file)
        print(f"[+] Top Disease: {disease}")
        print(f"[+] Top Targets: {', '.join(targets)}")
        
        topic = construct_research_topic(disease, targets)
        print(f"\n[+] Constructed Topic:\n{topic}\n")
        
        if args.dry_run:
            print("[+] Dry run complete. Exiting.")
            return
            
        success = run_autoresearchclaw(topic)
        if not success:
            print("[-] Aborting publication due to AutoResearchClaw failure.")
            return
            
        deliverables_dir = find_deliverables_folder()
        print(f"[+] Found generated deliverables at: {deliverables_dir}")
        
        publish_to_github(deliverables_dir, args.repo_url)
        
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()

"""
Phase 3: Client Simulation
A script simulating a paying biotech client ordering a research paper from our API.
"""

import requests
import time
import sys

API_URL = "http://127.0.0.1:8000"

def order_research(topic: str, disease: str):
    print(f"📡 Generating Research Order: {topic}")
    
    payload = {
        "client_topic": topic,
        "disease_target": disease
    }
    
    try:
        response = requests.post(f"{API_URL}/generate_research", json=payload)
        response.raise_for_status()
        data = response.json()
        
        job_id = data.get("job_id")
        print(f"✅ Order Accepted! Job ID: {job_id}")
        return job_id
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to {API_URL}. Is the FastAPI server running?")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def poll_status(job_id: str):
    """"Polls the API until the Nemotron research generation is complete."""
    print(f"\n⏳ Polling status for Job {job_id}...")
    
    while True:
        try:
            response = requests.get(f"{API_URL}/status/{job_id}")
            data = response.json()
            status = data.get("status")
            
            if status == "processing":
                print("   [⏳ processing] Nemotron LLM is running the simulation & drafting paper...")
                time.sleep(5) 
            elif status == "completed":
                print("\n🎉 Research Generation Complete!")
                print(f"📥 Download Link: {API_URL}{data.get('download_url')}")
                print("\n--- PREVIEW LOG HEADERS (Mock) ---")
                print("1. Abstract")
                print("2. 15D ODE Methodology")
                print("3. Phi-Complexity Results")
                print("4. Conclusion")
                break
            elif status == "failed":
                print(f"\n❌ Research Generation Failed: {data.get('error')}")
                break
            else:
                print(f"   [{status}] Waiting...")
                time.sleep(2)
                
        except Exception as e:
            print(f"Polling error: {e}")
            break

if __name__ == "__main__":
    test_topic = "Impact of glutamine deprivation on proliferative metabolism"
    target = "TNBC"
    
    print("==============================================")
    print(" PROJECT CONFLUENCE SAAS CLIENT PROXY")
    print("==============================================")
    
    job_id = order_research(test_topic, target)
    
    # In a real environment, this process might take 15 minutes.
    # We poll just to show the API flow.
    # Note: To see the full success, run `uvicorn app:app --reload` in `api_service/` first.
    poll_status(job_id)


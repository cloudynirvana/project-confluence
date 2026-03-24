"""
Phase 1: Biological Alpha Generator
Scrapes clinical trial/drug news and computationally simulates true efficacy using Confluence ODEs.
"""

import sys
import os
import json
import argparse
from duckduckgo_search import DDGS

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ode_system import ComplexAttractorODE, TNBCParams, GlioblastomaParams, NephroblastomaParams
from models.complexity_profiler import ComplexityProfiler

def gather_trial_news(query: str):
    """Scrapes recent news for the target drug/trial."""
    print(f"🔍 Searching for clinical trial news: {query}")
    results_text = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query + " clinical trial phase efficacy", max_results=3, timelimit="w")
            for r in results:
                results_text.append(r.get("body", ""))
    except Exception as e:
        print(f"Search failed: {e}")
    return " ".join(results_text)

def simulate_efficacy(disease_type: str, intervention_strength: float):
    """Simulates the drug's impact on the Complex Attractor ODE."""
    print(f"🧬 Simulating {disease_type} with drug strength {intervention_strength:.2f}...")
    
    # Map to params
    if disease_type.upper() == "TNBC":
        params = TNBCParams()
        params.glucose_uptake = params.glucose_uptake * (1.0 - intervention_strength) # Mock mechanism
    elif disease_type.upper() == "GBM":
        params = GlioblastomaParams()
        params.glycolysis_flux = params.glycolysis_flux * (1.0 - intervention_strength)
    else:
        params = NephroblastomaParams()
        params.IGF2_signaling = params.IGF2_signaling * (1.0 - intervention_strength)

    ode = ComplexAttractorODE(params=params, use_nonlinear=True, use_immune=True, use_microenv=True)
    profiler = ComplexityProfiler()
    
    sol = ode.solve(t_span=(0, 100), dt_eval=0.5)
    phi = profiler.profile(sol["z"], dt=0.5)
    
    return {
        "phi_magnitude": round(phi.phi_magnitude, 4),
        "coherence": round(phi.coherence_C, 4),
        "disease_archetype": phi.archetype
    }

def generate_alpha(target: str, disease: str):
    news = gather_trial_news(target)
    
    # Proxy: A real system would use Nemotron to extract exact metabolic shifts from the parsed news.
    intervention_strength = 0.50 
    
    sim_result = simulate_efficacy(disease, intervention_strength)
    
    alpha_payload = {
        "asset": target,
        "disease_target": disease,
        "market_context": news if news else "No recent news found.",
        "simulated_biological_reality": sim_result,
        "edge_summary": f"Biological simulation shows Phi magnitude of {sim_result['phi_magnitude']} and coherence {sim_result['coherence']}."
    }
    
    # Save for Phase 2
    os.makedirs("results/alpha", exist_ok=True)
    filepath = "results/alpha/latest_alpha.json"
    with open(filepath, "w") as f:
        json.dump(alpha_payload, f, indent=2)
        
    print(f"✅ Biological Alpha generated and saved to {filepath}")
    return alpha_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="Drug or company name (e.g. 'Keytruda', 'CRSP')")
    parser.add_argument("--disease", type=str, default="TNBC", help="Target disease model (TNBC, GBM, Nephroblastoma)")
    args = parser.parse_args()
    
    generate_alpha(args.target, args.disease)

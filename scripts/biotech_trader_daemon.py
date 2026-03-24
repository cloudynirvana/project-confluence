"""
Phase 3: Biotech Trading Daemon (ODE + MiroFish)
The main loop that orchestrates Phase 1 and Phase 2, and executes trades based on the divergence.
"""

import time
import os
import json
from biotech_alpha_generator import generate_alpha
from mirofish_market_simulator import run_market_simulation

def execute_trade(consensus: dict):
    """Executes the trade based on the MiroFish consensus edge."""
    action = consensus.get("recommended_action")
    asset = consensus.get("asset")
    confidence = consensus.get("confidence", 0.0)
    
    if confidence < 0.70 or action == "Neutral":
        print(f"⚖️ Edge too small or neutral. No trade executed for {asset}.")
        return

    print(f"💰 EXECUTING TRADE: {action} {asset} with {confidence*100:.1f}% confidence.")
    print(f"📝 Rationale: {consensus.get('rationale')}")
    
    # Example logic using polymarket_mcp (mocked for print)
    if action == "SHORT" or action == "SELL":
        print(f"[Polymarket Client] --> Selling YES shares / Buying NO shares for {asset}")
    else:
        print(f"[Polymarket Client] --> Buying YES shares for {asset}")
        
    print("✅ Trade completed.")

def run_trading_cycle(target: str, disease: str):
    print(f"\n{'='*50}")
    print(f"🚀 INITIATING BIOTECH GOD-VIEW CYCLE: {target} ({disease})")
    print(f"{'='*50}\n")
    
    # 1. Biological Alpha
    alpha = generate_alpha(target, disease)
    
    # 2. MiroFish Market Simulation
    alpha_path = "results/alpha/latest_alpha.json"
    consensus = run_market_simulation(alpha_path)
    
    if not consensus:
        print("Simulation failed. Aborting trade cycle.")
        return
        
    # 3. Trade Execution
    print("\n" + "-"*30)
    execute_trade(consensus)
    print("-"*30 + "\n")

if __name__ == "__main__":
    # Example watch list 
    watch_list = [
        {"target": "CRSP Sickle Cell Trial", "disease": "TNBC"}, 
        {"target": "VRTX Phase 3 Efficacy", "disease": "Nephroblastoma"} 
    ]
    
    for item in watch_list:
        run_trading_cycle(item["target"], item["disease"])
        time.sleep(5)

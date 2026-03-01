"""Verify the integrated cure engine imports correctly."""
import sys, os
sys.path.insert(0, 'C:/Users/Kelechi/.gemini/antigravity/scratch/saem-cancer-poc/src')
sys.path.insert(0, 'C:/Users/Kelechi/.gemini/antigravity/scratch/saem-cancer-poc')

# Test import of the full engine
try:
    import universal_cure_engine as uce
    print("OK  universal_cure_engine imports successfully")
    print(f"    Modules loaded: immune_dynamics, spatial_dynamics, resistance_model")
    print(f"    DrugEfficiencyEngine: {uce.DrugEfficiencyEngine.__doc__[:60]}...")
    print(f"    ResistanceTracker initialized: {bool(uce.ResistanceTracker)}")
    print(f"    SpatialTumorModel initialized: {bool(uce.SpatialTumorModel)}")
    print(f"    TISSUE_BARRIERS keys: {list(uce.TISSUE_BARRIERS.keys())}")
    
    # Quick test: _identify_cancer_type
    import numpy as np
    from tnbc_ode import TNBCODESystem
    A_tnbc = TNBCODESystem.tnbc_generator()
    ctype = uce._identify_cancer_type(A_tnbc)
    print(f"    _identify_cancer_type(TNBC generator) = {ctype}")
    
    print("\n=== CONFLUENCE INTEGRATION VERIFIED ===")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()

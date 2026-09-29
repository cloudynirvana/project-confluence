# Limitations

- Simulated scenarios, not real patients or cells (unless a section says otherwise).
- Parameters marked "assumed" have not been fitted to data.
- Model timescale and scale are stated per model; do not extrapolate to tumours or clinical stages.
- Resistance mechanisms may be missing.
- The DOI previously listed in `CITATION.cff` (`10.5281/zenodo.21446803`) did not resolve on 2026-09-29. Cite the GitHub repository until a Zenodo record exists.
- Structural identifiability rising from 7/17 to 15/17 is stated in the README and has not been independently re-run in this hygiene pass.
- What would falsify this:
  - On the same simulated uncertainty set, if the adaptive policy does not reduce resistant takeover relative to the MTD comparator.
  - If the 15/17 identifiability count cannot be reproduced on the six stated metabolomics channels.
- Validation path: model -> public-data plausibility -> wet-lab -> animals -> trials.

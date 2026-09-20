# OnCo adapter (P0)

Read-only client for [OnCo](https://onco.cc). Does **not** modify `CancerODE.rhs_cancer`.

OnCo data is CC BY-NC 4.0. Adapter code in this repository is MIT, same as the rest of Confluence. Cached records stay OnCo data; do not vendor `all.json`.

```
Knowledge ≠ Evidence ≠ Causal mechanism ≠ Parameter ≠ Prediction
legacy gene→parameter map ≠ identified parameter mapping
```

## Layout

- `confluence/onco/` — client, cache envelope, schemas, bindings
- `docs/ONCO_CONFLUENCE_ONTOLOGY_SPEC.md` — v0.3 spec
- `data/onco/fixtures/` — offline fixtures only
- `data/onco/cache/` — gitignored live cache
- `tests/test_onco_adapter.py` — nine gates
- `confluence/profiles/` — Disease Profile export (thinking lab). Does not write Θ.
- `docs/CITATION_POLICY.md` — public-claim citation rules

## Commands

```bash
python -m confluence.onco --api data/onco/fixtures meta
python -m confluence.onco bind --id ldha
python -m confluence.onco wired
ONCO_API=https://onco.cc/api/v1 python -m confluence.onco meta
```

`bind()` returns slot annotations. It never writes `p_lactate` or `pyruvate_to_lactate`.
`validation/gene_to_parameter_map.json` is classified `legacy_mapping` / `assumed` / `unidentified`.

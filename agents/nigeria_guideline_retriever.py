"""
Nigeria Guideline Retriever — Project Confluence
===================================================

RAG (Retrieval-Augmented Generation) module over the
Nigeria Standard Treatment Guidelines (NSTG 2022).

Loads the structured clinical guidelines dataset (270 conditions)
and provides semantic search for guideline-compliant clinical queries.

Data Source:
    HuggingFace: chisomrutherford/nigeria-clinical-guidelines-dataset
    License: CC-BY-4.0

Pipeline Position:
    Clinical Query → [NigeriaGuidelineRetriever] → Guideline Passages
                                                     ↓
    AdaptiveController ← dosing constraints ← extract_constraints()
    PatientFitter      ← prior bounds       ← get_treatment_protocol()
    API Service        ← /guideline_query    ← retrieve() / answer()

References:
    Federal Ministry of Health, Nigeria (2022)
    Nigeria Standard Treatment Guidelines (3rd Edition)
"""

import json
import os
import re
import warnings
import hashlib
import pickle
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ── Optional dependency imports (graceful fallback) ─────────────────────
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GuidelineChunk:
    """A single retrievable chunk from the NSTG 2022 guidelines."""
    condition_name: str
    field_type: str          # e.g., "introduction", "treatment_protocols", etc.
    text: str                # The flattened text content
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "condition_name": self.condition_name,
            "field_type": self.field_type,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """A single retrieval result with score and source chunk."""
    chunk: GuidelineChunk
    score: float
    rank: int

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "condition": self.chunk.condition_name,
            "field_type": self.chunk.field_type,
            "text": self.chunk.text[:500],
            "metadata": self.chunk.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════

class NigeriaGuidelineRetriever:
    """
    RAG retriever over Nigeria Standard Treatment Guidelines (NSTG 2022).

    Loads 270 clinical conditions from the HuggingFace dataset,
    chunks them at field level, and provides semantic search via
    sentence-transformers + FAISS (or TF-IDF fallback).

    Usage:
        retriever = NigeriaGuidelineRetriever()
        results = retriever.retrieve("first-line treatment for breast cancer")
        for r in results:
            print(f"[{r.score:.3f}] {r.chunk.condition_name}: {r.chunk.text[:100]}")

        # Direct protocol lookup
        protocol = retriever.get_treatment_protocol("BREAST CANCER")

        # Structured clinical answer
        answer = retriever.answer("What are the dosing guidelines for doxorubicin?")
    """

    # Default embedding model (80MB, CPU-friendly)
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Cache directory (relative to project root)
    CACHE_DIR = "data/nigeria_guidelines_index"

    def __init__(
        self,
        data_source: str = "chisomrutherford/nigeria-clinical-guidelines-dataset",
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        local_data_path: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        """
        Initialize the retriever.

        Args:
            data_source: HuggingFace dataset ID.
            model_name: Sentence-transformers model for embeddings.
            cache_dir: Directory to cache the FAISS index.
            local_data_path: Path to local JSON files (bypass HF download).
            force_rebuild: Force rebuild of the index even if cache exists.
        """
        self.data_source = data_source
        self.model_name = model_name
        self.local_data_path = local_data_path

        # Resolve cache directory
        project_root = Path(__file__).resolve().parent.parent
        self.cache_dir = Path(cache_dir) if cache_dir else project_root / self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self.chunks: List[GuidelineChunk] = []
        self.conditions: Dict[str, Dict] = {}  # condition_name → raw JSON
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._model = None
        self._use_faiss = HAS_SENTENCE_TRANSFORMERS and HAS_FAISS
        self._is_loaded = False

        # Load data and build index
        self._load_data()
        self._build_index(force_rebuild=force_rebuild)

    # ── Data Loading ──────────────────────────────────────────────────────

    def _load_data(self):
        """Load the NSTG 2022 dataset and build chunks."""
        raw_conditions = self._fetch_raw_data()
        self.conditions = {}
        self.chunks = []

        for condition_data in raw_conditions:
            name = condition_data.get("condition_name", "Unknown")
            self.conditions[name.upper()] = condition_data
            self._chunk_condition(condition_data)

        self._is_loaded = True
        print(f"[NigeriaGuidelineRetriever] Loaded {len(self.conditions)} conditions, "
              f"{len(self.chunks)} chunks")

    def _fetch_raw_data(self) -> List[Dict]:
        """Fetch raw condition JSON data from HuggingFace or local path."""
        # 1. Try local path first
        if self.local_data_path:
            return self._load_local_json(self.local_data_path)

        # 2. Try HuggingFace datasets library
        if HAS_DATASETS:
            try:
                ds = load_dataset(self.data_source, trust_remote_code=True)
                # The dataset may be a dict of splits or a flat dataset
                if hasattr(ds, 'keys'):
                    split = list(ds.keys())[0]
                    return [dict(row) for row in ds[split]]
                return [dict(row) for row in ds]
            except Exception as e:
                warnings.warn(f"HuggingFace datasets load failed: {e}. Trying local fallback.")

        # 3. Try cached local copy
        local_cache = self.cache_dir / "raw_conditions.json"
        if local_cache.exists():
            with open(local_cache) as f:
                return json.load(f)

        # 4. Generate mock data for offline development
        warnings.warn(
            "Cannot load Nigeria Guidelines dataset. "
            "Install 'datasets' package or provide local_data_path. "
            "Using built-in mock data for development."
        )
        return self._mock_conditions()

    def _load_local_json(self, path: str) -> List[Dict]:
        """Load condition JSON files from a local directory."""
        p = Path(path)
        conditions = []
        if p.is_dir():
            for f in sorted(p.glob("*.json")):
                with open(f) as fh:
                    conditions.append(json.load(fh))
        elif p.is_file():
            with open(p) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    conditions = data
                else:
                    conditions = [data]
        return conditions

    def _chunk_condition(self, condition: Dict):
        """Break a condition JSON into retrievable chunks."""
        name = condition.get("condition_name", "Unknown")
        slug = condition.get("condition_slug", "")

        base_meta = {"source": condition.get("source", "NSTG 2022"), "slug": slug}

        # Introduction
        intro = condition.get("introduction", "")
        if intro:
            self.chunks.append(GuidelineChunk(
                condition_name=name,
                field_type="introduction",
                text=f"{name}: {intro}",
                metadata=base_meta,
            ))

        # Clinical features
        clinical_features = condition.get("clinical_features", [])
        if clinical_features:
            text_parts = []
            for cf in clinical_features:
                if isinstance(cf, dict):
                    cf_type = cf.get("type", "")
                    features = cf.get("features", [])
                    if cf_type:
                        text_parts.append(f"{cf_type}:")
                    text_parts.extend(
                        f for f in features if isinstance(f, str)
                    )
                elif isinstance(cf, str):
                    text_parts.append(cf)
            if text_parts:
                self.chunks.append(GuidelineChunk(
                    condition_name=name,
                    field_type="clinical_features",
                    text=f"{name} — Clinical Features: " + " | ".join(text_parts),
                    metadata=base_meta,
                ))

        # Investigations
        investigations = condition.get("investigations", [])
        if investigations:
            inv_text = [i for i in investigations if isinstance(i, str)]
            if inv_text:
                self.chunks.append(GuidelineChunk(
                    condition_name=name,
                    field_type="investigations",
                    text=f"{name} — Investigations: " + " | ".join(inv_text),
                    metadata=base_meta,
                ))

        # Treatment protocols (the most valuable field)
        treatment_protocols = condition.get("treatment_protocols", [])
        for tp in treatment_protocols:
            if isinstance(tp, dict):
                tp_type = tp.get("type", "Treatment")
                parts = [f"{name} — {tp_type}:"]

                # Drugs
                drugs = tp.get("drug", tp.get("drugs", []))
                if isinstance(drugs, list):
                    for drug in drugs:
                        if isinstance(drug, dict):
                            drug_name = drug.get("name", drug.get("drug", ""))
                            dosage = drug.get("dosage", drug.get("dose", ""))
                            route = drug.get("route", "")
                            frequency = drug.get("frequency", "")
                            duration = drug.get("duration", "")
                            parts.append(
                                f"Drug: {drug_name}, Dosage: {dosage}, "
                                f"Route: {route}, Frequency: {frequency}, "
                                f"Duration: {duration}".strip(", ")
                            )
                        elif isinstance(drug, str):
                            parts.append(f"Drug: {drug}")

                # Adverse reactions and cautions
                adverse = tp.get("adverse_reactions_and_cautions", [])
                if adverse:
                    if isinstance(adverse, list):
                        parts.append("Adverse reactions/cautions: " +
                                     " | ".join(str(a) for a in adverse))
                    elif isinstance(adverse, str):
                        parts.append(f"Adverse reactions/cautions: {adverse}")

                # Supportive measures
                supportive = tp.get("supportive_measures", [])
                if supportive:
                    if isinstance(supportive, list):
                        parts.append("Supportive measures: " +
                                     " | ".join(str(s) for s in supportive))

                text = " ".join(parts)
                if len(text) > 50:  # Skip near-empty chunks
                    self.chunks.append(GuidelineChunk(
                        condition_name=name,
                        field_type="treatment_protocol",
                        text=text,
                        metadata={**base_meta, "protocol_type": tp_type},
                    ))
            elif isinstance(tp, str):
                self.chunks.append(GuidelineChunk(
                    condition_name=name,
                    field_type="treatment_protocol",
                    text=f"{name} — Treatment: {tp}",
                    metadata=base_meta,
                ))

        # Differential diagnoses
        diff_dx = condition.get("differential_diagnoses", [])
        if diff_dx:
            dx_text = [d for d in diff_dx if isinstance(d, str)]
            if dx_text:
                self.chunks.append(GuidelineChunk(
                    condition_name=name,
                    field_type="differential_diagnoses",
                    text=f"{name} — Differential Diagnoses: " + " | ".join(dx_text),
                    metadata=base_meta,
                ))

        # Complications
        complications = condition.get("complications", [])
        if complications:
            comp_text = [c for c in complications if isinstance(c, str)]
            if comp_text:
                self.chunks.append(GuidelineChunk(
                    condition_name=name,
                    field_type="complications",
                    text=f"{name} — Complications: " + " | ".join(comp_text),
                    metadata=base_meta,
                ))

        # Prevention
        prevention = condition.get("prevention", [])
        if prevention:
            prev_text = [p for p in prevention if isinstance(p, str)]
            if prev_text:
                self.chunks.append(GuidelineChunk(
                    condition_name=name,
                    field_type="prevention",
                    text=f"{name} — Prevention: " + " | ".join(prev_text),
                    metadata=base_meta,
                ))

        # Prognosis
        prognosis = condition.get("prognosis", [])
        if prognosis:
            prog_text = [p for p in prognosis if isinstance(p, str)]
            if prog_text:
                self.chunks.append(GuidelineChunk(
                    condition_name=name,
                    field_type="prognosis",
                    text=f"{name} — Prognosis: " + " | ".join(prog_text),
                    metadata=base_meta,
                ))

    # ── Index Building ────────────────────────────────────────────────────

    def _build_index(self, force_rebuild: bool = False):
        """Build the search index (FAISS or TF-IDF fallback)."""
        if not self.chunks:
            warnings.warn("No chunks to index.")
            return

        if self._use_faiss:
            self._build_faiss_index(force_rebuild)
        elif HAS_SKLEARN:
            self._build_tfidf_index()
        else:
            warnings.warn(
                "Neither sentence-transformers+FAISS nor scikit-learn available. "
                "Search will use basic string matching."
            )

    def _get_cache_hash(self) -> str:
        """Compute a hash of the current chunks for cache validity."""
        content = "|".join(c.text[:50] for c in self.chunks[:20])
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _build_faiss_index(self, force_rebuild: bool = False):
        """Build FAISS index with sentence-transformer embeddings."""
        cache_hash = self._get_cache_hash()
        index_path = self.cache_dir / f"faiss_index_{cache_hash}.bin"
        embeddings_path = self.cache_dir / f"embeddings_{cache_hash}.npy"

        # Try loading from cache
        if not force_rebuild and index_path.exists() and embeddings_path.exists():
            try:
                self._faiss_index = faiss.read_index(str(index_path))
                self._embeddings = np.load(str(embeddings_path))
                print(f"[NigeriaGuidelineRetriever] Loaded cached FAISS index "
                      f"({self._embeddings.shape[0]} vectors)")
                return
            except Exception as e:
                warnings.warn(f"Cache load failed: {e}. Rebuilding.")

        # Build from scratch
        print(f"[NigeriaGuidelineRetriever] Building FAISS index with "
              f"{self.model_name}...")
        self._model = SentenceTransformer(self.model_name)

        texts = [c.text for c in self.chunks]
        self._embeddings = self._model.encode(
            texts, show_progress_bar=True, batch_size=32
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(self._embeddings)

        # Build index
        dim = self._embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized
        self._faiss_index.add(self._embeddings)

        # Cache to disk
        try:
            faiss.write_index(self._faiss_index, str(index_path))
            np.save(str(embeddings_path), self._embeddings)
            print(f"[NigeriaGuidelineRetriever] Cached FAISS index to {index_path}")
        except Exception as e:
            warnings.warn(f"Failed to cache index: {e}")

        print(f"[NigeriaGuidelineRetriever] FAISS index built: "
              f"{self._embeddings.shape[0]} vectors, dim={dim}")

    def _build_tfidf_index(self):
        """Fallback: build TF-IDF index."""
        print("[NigeriaGuidelineRetriever] Building TF-IDF fallback index...")
        texts = [c.text for c in self.chunks]
        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
        print(f"[NigeriaGuidelineRetriever] TF-IDF index built: "
              f"{self._tfidf_matrix.shape}")

    # ── Retrieval Methods ─────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5,
                 field_filter: Optional[str] = None) -> List[RetrievalResult]:
        """
        Semantic search over NSTG 2022 guidelines.

        Args:
            query: Natural language clinical query.
            top_k: Number of top results to return.
            field_filter: Optional filter by field_type (e.g., "treatment_protocol").

        Returns:
            List of RetrievalResult sorted by relevance score.
        """
        if not self.chunks:
            return []

        if self._use_faiss and self._faiss_index is not None:
            return self._retrieve_faiss(query, top_k, field_filter)
        elif self._tfidf_matrix is not None:
            return self._retrieve_tfidf(query, top_k, field_filter)
        else:
            return self._retrieve_keyword(query, top_k, field_filter)

    def _retrieve_faiss(self, query: str, top_k: int,
                        field_filter: Optional[str]) -> List[RetrievalResult]:
        """Retrieve using FAISS + sentence-transformers."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        query_vec = self._model.encode([query])
        faiss.normalize_L2(query_vec)

        # Search more than top_k to allow for filtering
        search_k = top_k * 3 if field_filter else top_k
        scores, indices = self._faiss_index.search(query_vec, min(search_k, len(self.chunks)))

        results = []
        for rank_idx, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if field_filter and chunk.field_type != field_filter:
                continue
            results.append(RetrievalResult(
                chunk=chunk, score=float(score), rank=len(results) + 1
            ))
            if len(results) >= top_k:
                break

        return results

    def _retrieve_tfidf(self, query: str, top_k: int,
                        field_filter: Optional[str]) -> List[RetrievalResult]:
        """Retrieve using TF-IDF + cosine similarity."""
        query_vec = self._tfidf_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Sort by similarity
        top_indices = np.argsort(sims)[::-1]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            if field_filter and chunk.field_type != field_filter:
                continue
            if sims[idx] <= 0:
                continue
            results.append(RetrievalResult(
                chunk=chunk, score=float(sims[idx]), rank=len(results) + 1
            ))
            if len(results) >= top_k:
                break

        return results

    def _retrieve_keyword(self, query: str, top_k: int,
                          field_filter: Optional[str]) -> List[RetrievalResult]:
        """Basic keyword matching fallback."""
        query_terms = set(query.lower().split())
        scored = []

        for chunk in self.chunks:
            if field_filter and chunk.field_type != field_filter:
                continue
            chunk_terms = set(chunk.text.lower().split())
            overlap = len(query_terms & chunk_terms)
            if overlap > 0:
                score = overlap / max(len(query_terms), 1)
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(chunk=c, score=s, rank=i + 1)
            for i, (c, s) in enumerate(scored[:top_k])
        ]

    # ── High-Level Query Methods ──────────────────────────────────────────

    def answer(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve guideline passages and format a structured clinical answer.

        Args:
            query: Natural language clinical question.
            top_k: Number of source passages to include.

        Returns:
            Formatted answer string with source citations.
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return (
                f"No relevant Nigerian clinical guidelines found for: '{query}'\n"
                f"The NSTG 2022 dataset contains {len(self.conditions)} conditions. "
                f"Try rephrasing your query."
            )

        lines = [
            f"═══ Nigeria Standard Treatment Guidelines (NSTG 2022) ═══",
            f"Query: {query}",
            f"Sources: {len(results)} guideline passages retrieved",
            "",
        ]

        for r in results:
            lines.append(f"── [{r.rank}] {r.chunk.condition_name} "
                         f"({r.chunk.field_type}) — Score: {r.score:.3f} ──")
            lines.append(r.chunk.text)
            lines.append("")

        lines.append(f"═══ Source: Federal Ministry of Health, Nigeria (2022) ═══")
        return "\n".join(lines)

    def get_treatment_protocol(self, condition: str) -> Optional[Dict]:
        """
        Direct lookup: returns the full structured treatment protocol
        for an exact condition name.

        Args:
            condition: Condition name (case-insensitive).

        Returns:
            The raw condition JSON dict, or None if not found.
        """
        key = condition.upper().strip()

        # Exact match
        if key in self.conditions:
            return self.conditions[key]

        # Fuzzy match (substring)
        for cond_name, data in self.conditions.items():
            if key in cond_name or cond_name in key:
                return data

        return None

    def get_dosing_constraints(self, drug_name: str) -> List[Dict]:
        """
        Extract dosing limits, adverse reactions, and cautions for a
        specific drug across all conditions that mention it.

        Args:
            drug_name: Drug name to search for (case-insensitive).

        Returns:
            List of dicts with condition, dosage, route, adverse_reactions.
        """
        drug_lower = drug_name.lower()
        results = []

        for cond_name, condition in self.conditions.items():
            protocols = condition.get("treatment_protocols", [])
            for tp in protocols:
                if not isinstance(tp, dict):
                    continue
                drugs = tp.get("drug", tp.get("drugs", []))
                if not isinstance(drugs, list):
                    continue
                for drug in drugs:
                    if isinstance(drug, dict):
                        name = drug.get("name", drug.get("drug", "")).lower()
                        if drug_lower in name or name in drug_lower:
                            results.append({
                                "condition": cond_name,
                                "protocol_type": tp.get("type", ""),
                                "drug_name": drug.get("name", drug.get("drug", "")),
                                "dosage": drug.get("dosage", drug.get("dose", "")),
                                "route": drug.get("route", ""),
                                "frequency": drug.get("frequency", ""),
                                "duration": drug.get("duration", ""),
                                "adverse_reactions": tp.get(
                                    "adverse_reactions_and_cautions", []),
                            })
                    elif isinstance(drug, str) and drug_lower in drug.lower():
                        results.append({
                            "condition": cond_name,
                            "protocol_type": tp.get("type", ""),
                            "drug_name": drug,
                            "adverse_reactions": tp.get(
                                "adverse_reactions_and_cautions", []),
                        })

        return results

    def extract_oncology_constraints(self) -> Dict:
        """
        Extract oncology-specific constraints from the guidelines for
        use by the AdaptiveController's safety layer.

        Returns:
            Dict of cancer-related constraints organized by condition.
        """
        oncology_keywords = [
            "cancer", "carcinoma", "tumour", "tumor", "leukaemia", "leukemia",
            "lymphoma", "sarcoma", "melanoma", "neoplasm", "malignant",
            "breast", "cervical", "prostate", "colorectal", "lung",
            "chemotherapy", "radiation",
        ]

        constraints = {}
        for cond_name, condition in self.conditions.items():
            # Check if condition is oncology-related
            intro = condition.get("introduction", "").lower()
            name_lower = cond_name.lower()
            is_oncology = any(
                kw in name_lower or kw in intro
                for kw in oncology_keywords
            )
            if not is_oncology:
                continue

            protocols = condition.get("treatment_protocols", [])
            drug_list = []
            adverse_reactions = []
            for tp in protocols:
                if not isinstance(tp, dict):
                    continue
                drugs = tp.get("drug", tp.get("drugs", []))
                if isinstance(drugs, list):
                    for d in drugs:
                        if isinstance(d, dict):
                            drug_list.append({
                                "name": d.get("name", d.get("drug", "")),
                                "dosage": d.get("dosage", d.get("dose", "")),
                                "route": d.get("route", ""),
                            })
                ar = tp.get("adverse_reactions_and_cautions", [])
                if isinstance(ar, list):
                    adverse_reactions.extend(ar)
                elif isinstance(ar, str):
                    adverse_reactions.append(ar)

            constraints[cond_name] = {
                "drugs": drug_list,
                "adverse_reactions": adverse_reactions,
                "investigations": condition.get("investigations", []),
                "complications": condition.get("complications", []),
                "prevention": condition.get("prevention", []),
            }

        return constraints

    # ── Utility ───────────────────────────────────────────────────────────

    def list_conditions(self) -> List[str]:
        """Return all 270 condition names."""
        return sorted(self.conditions.keys())

    def get_stats(self) -> Dict:
        """Return retriever statistics."""
        return {
            "n_conditions": len(self.conditions),
            "n_chunks": len(self.chunks),
            "index_type": "FAISS" if self._use_faiss else (
                "TF-IDF" if self._tfidf_matrix is not None else "keyword"),
            "model": self.model_name if self._use_faiss else "N/A",
            "source": self.data_source,
            "field_types": list(set(c.field_type for c in self.chunks)),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"NigeriaGuidelineRetriever("
                f"conditions={stats['n_conditions']}, "
                f"chunks={stats['n_chunks']}, "
                f"index={stats['index_type']})")

    # ── Mock Data (offline development) ───────────────────────────────────

    def _mock_conditions(self) -> List[Dict]:
        """Built-in mock conditions for offline testing."""
        return [
            {
                "condition_name": "BREAST CANCER",
                "condition_slug": "breast-cancer",
                "source": "NSTG 2022 (Mock)",
                "introduction": "Breast cancer is the most common cancer in Nigerian women. It often presents at advanced stages due to late presentation and limited screening infrastructure.",
                "clinical_features": [
                    {"type": "Symptoms", "features": [
                        "Painless breast lump", "Nipple discharge",
                        "Skin changes (peau d'orange)", "Axillary lymphadenopathy"
                    ]}
                ],
                "investigations": [
                    "Fine needle aspiration cytology (FNAC)",
                    "Core needle biopsy", "Mammography",
                    "Breast ultrasound", "Chest X-ray",
                    "Abdominal ultrasound", "Full blood count",
                    "Liver function tests", "Hormonal receptor status (ER/PR/HER2)"
                ],
                "treatment_protocols": [
                    {
                        "type": "Primary Care — Early Stage",
                        "drug": [
                            {"name": "Tamoxifen", "dosage": "20mg", "route": "oral",
                             "frequency": "once daily", "duration": "5 years"},
                        ],
                        "adverse_reactions_and_cautions": [
                            "Hot flushes", "Risk of endometrial cancer",
                            "Thromboembolic events", "Monitor with regular gynaecological exams"
                        ],
                        "supportive_measures": [
                            "Nutritional support", "Psychosocial counselling",
                            "Pain management"
                        ]
                    },
                    {
                        "type": "Chemotherapy — Locally Advanced/Metastatic",
                        "drug": [
                            {"name": "Doxorubicin", "dosage": "60mg/m²", "route": "IV",
                             "frequency": "every 3 weeks", "duration": "4-6 cycles"},
                            {"name": "Cyclophosphamide", "dosage": "600mg/m²", "route": "IV",
                             "frequency": "every 3 weeks", "duration": "4-6 cycles"},
                            {"name": "5-Fluorouracil", "dosage": "500mg/m²", "route": "IV",
                             "frequency": "every 3 weeks", "duration": "4-6 cycles"},
                        ],
                        "adverse_reactions_and_cautions": [
                            "Myelosuppression — monitor FBC regularly",
                            "Cardiotoxicity (doxorubicin) — cumulative dose limit 450mg/m²",
                            "Nausea and vomiting — antiemetic prophylaxis required",
                            "Alopecia", "Mucositis",
                            "Neutropenic fever — emergency management required"
                        ]
                    }
                ],
                "differential_diagnoses": [
                    "Fibroadenoma", "Breast cyst", "Fat necrosis",
                    "Phyllodes tumour", "Mastitis"
                ],
                "complications": [
                    "Metastatic disease", "Lymphoedema",
                    "Chemotherapy-induced cardiomyopathy"
                ],
                "prevention": [
                    "Breast self-examination", "Clinical breast examination",
                    "Mammographic screening where available"
                ],
            },
            {
                "condition_name": "MALARIA",
                "condition_slug": "malaria",
                "source": "NSTG 2022 (Mock)",
                "introduction": "Malaria is endemic in Nigeria and is a major cause of morbidity and mortality. Plasmodium falciparum is the predominant species.",
                "clinical_features": [
                    {"type": "Symptoms", "features": [
                        "Fever", "Chills and rigors", "Headache",
                        "Body aches", "Nausea/vomiting"
                    ]}
                ],
                "investigations": [
                    "Malaria rapid diagnostic test (mRDT)",
                    "Thick and thin blood film", "Full blood count",
                    "Blood glucose", "Serum electrolytes"
                ],
                "treatment_protocols": [
                    {
                        "type": "Uncomplicated Malaria",
                        "drug": [
                            {"name": "Artemether-Lumefantrine", "dosage": "Weight-based",
                             "route": "oral", "frequency": "twice daily",
                             "duration": "3 days"},
                        ],
                        "adverse_reactions_and_cautions": [
                            "GI disturbance", "Dizziness",
                            "Avoid in first trimester of pregnancy"
                        ]
                    }
                ],
                "complications": [
                    "Severe malaria", "Cerebral malaria", "Severe anaemia",
                    "Acute kidney injury", "Hypoglycaemia"
                ],
                "prevention": [
                    "Insecticide-treated nets (ITNs)",
                    "Indoor residual spraying (IRS)",
                    "Intermittent preventive therapy in pregnancy (IPTp)"
                ],
            },
            {
                "condition_name": "ANAEMIA",
                "condition_slug": "anaemia",
                "source": "NSTG 2022 (Mock)",
                "introduction": "Anaemia is common in Nigeria, particularly iron deficiency anaemia. It is a significant comorbidity in cancer patients undergoing chemotherapy.",
                "clinical_features": [
                    {"type": "Symptoms", "features": [
                        "Fatigue", "Pallor", "Dyspnoea on exertion",
                        "Palpitations", "Dizziness"
                    ]}
                ],
                "treatment_protocols": [
                    {
                        "type": "Iron Deficiency Anaemia",
                        "drug": [
                            {"name": "Ferrous sulphate", "dosage": "200mg",
                             "route": "oral", "frequency": "three times daily",
                             "duration": "3-6 months"},
                        ],
                        "adverse_reactions_and_cautions": [
                            "GI side effects (nausea, constipation)",
                            "Take with food if GI intolerance"
                        ]
                    }
                ],
                "complications": [
                    "Heart failure", "Impaired immune function"
                ],
            },
            {
                "condition_name": "SICKLE CELL DISEASE",
                "condition_slug": "sickle-cell-disease",
                "source": "NSTG 2022 (Mock)",
                "introduction": "Sickle cell disease is highly prevalent in Nigeria with approximately 2-3% of the population being homozygous (HbSS). It is a critical comorbidity consideration in oncology patients.",
                "clinical_features": [
                    {"type": "Symptoms", "features": [
                        "Painful crises", "Jaundice", "Chronic anaemia",
                        "Splenic sequestration", "Recurrent infections"
                    ]}
                ],
                "treatment_protocols": [
                    {
                        "type": "Crisis Management",
                        "drug": [
                            {"name": "Hydroxyurea", "dosage": "15-35mg/kg/day",
                             "route": "oral", "frequency": "once daily",
                             "duration": "long-term"},
                        ],
                        "adverse_reactions_and_cautions": [
                            "Myelosuppression — regular FBC monitoring",
                            "Contraindicated in pregnancy",
                            "Monitor renal and hepatic function"
                        ]
                    }
                ],
                "complications": [
                    "Stroke", "Acute chest syndrome",
                    "Aplastic crisis", "Osteonecrosis"
                ],
            },
            {
                "condition_name": "HIV/AIDS",
                "condition_slug": "hiv-aids",
                "source": "NSTG 2022 (Mock)",
                "introduction": "Nigeria has one of the largest HIV burdens globally. HIV co-infection significantly complicates cancer treatment, particularly for cervical cancer and Kaposi sarcoma.",
                "clinical_features": [
                    {"type": "Symptoms", "features": [
                        "Chronic fever", "Weight loss",
                        "Persistent diarrhoea", "Opportunistic infections"
                    ]}
                ],
                "treatment_protocols": [
                    {
                        "type": "First-Line ART",
                        "drug": [
                            {"name": "Tenofovir/Lamivudine/Dolutegravir (TLD)",
                             "dosage": "300/300/50mg", "route": "oral",
                             "frequency": "once daily", "duration": "lifelong"},
                        ],
                        "adverse_reactions_and_cautions": [
                            "Renal toxicity (tenofovir) — monitor creatinine",
                            "Weight gain (dolutegravir)",
                            "Drug interactions with rifampicin (TB co-treatment)"
                        ]
                    }
                ],
                "complications": [
                    "Opportunistic infections", "Immune reconstitution syndrome (IRIS)",
                    "HIV-associated malignancies"
                ],
            },
            {
                "condition_name": "CERVICAL CANCER",
                "condition_slug": "cervical-cancer",
                "source": "NSTG 2022 (Mock)",
                "introduction": "Cervical cancer is the second most common cancer in Nigerian women, closely linked to HPV infection and often presenting at advanced stages.",
                "clinical_features": [
                    {"type": "Symptoms", "features": [
                        "Abnormal vaginal bleeding", "Post-coital bleeding",
                        "Vaginal discharge", "Pelvic pain"
                    ]}
                ],
                "treatment_protocols": [
                    {
                        "type": "Locally Advanced",
                        "drug": [
                            {"name": "Cisplatin", "dosage": "40mg/m²", "route": "IV",
                             "frequency": "weekly", "duration": "5-6 weeks (concurrent with radiation)"},
                        ],
                        "adverse_reactions_and_cautions": [
                            "Nephrotoxicity — aggressive hydration required",
                            "Ototoxicity", "Peripheral neuropathy",
                            "Myelosuppression", "Nausea/vomiting"
                        ]
                    }
                ],
                "differential_diagnoses": [
                    "Cervicitis", "Cervical polyp", "Endometrial cancer"
                ],
                "prevention": [
                    "HPV vaccination", "Cervical screening (VIA, Pap smear)"
                ],
            },
        ]

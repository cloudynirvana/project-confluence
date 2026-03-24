"""
AlphaFold Client Module — Project Confluence
===============================================

REST client for the AlphaFold Protein Structure Database (alphafold.ebi.ac.uk).
Fetches predicted protein structures, extracts structural features, and
identifies binding pockets for drug-target analysis.

No heavyweight dependencies — uses requests (already in requirements.txt)
and biopython for mmCIF/PDB parsing.

References:
    Jumper et al. (2021) - Highly accurate protein structure prediction with AlphaFold
    Varadi et al. (2022) - AlphaFold Protein Structure Database
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.PDBParser import PDBParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ResidueInfo:
    """Per-residue structural information from AlphaFold prediction."""
    index: int              # 1-based residue index
    name: str               # 3-letter amino acid code
    plddt: float            # Predicted LDDT confidence (0-100)
    coords: np.ndarray      # Cα coordinates (x, y, z)
    secondary_structure: str = "C"  # H=helix, E=sheet, C=coil (estimated)


@dataclass
class BindingPocket:
    """A predicted binding pocket from geometric analysis."""
    center: np.ndarray         # Pocket centroid (x, y, z)
    residue_indices: List[int] # Residue indices forming the pocket
    volume: float              # Estimated pocket volume (Å³)
    avg_plddt: float           # Mean pLDDT of pocket residues
    depth: float               # Pocket depth estimate (Å)
    druggability_score: float  # 0-1 heuristic druggability


@dataclass
class StructureData:
    """Parsed AlphaFold structure with extracted features."""
    uniprot_id: str
    gene_name: str
    organism: str
    residues: List[ResidueInfo]
    pockets: List[BindingPocket] = field(default_factory=list)
    model_version: str = "alphafold_v4"
    source_url: str = ""

    @property
    def sequence_length(self) -> int:
        return len(self.residues)

    @property
    def mean_plddt(self) -> float:
        if not self.residues:
            return 0.0
        return float(np.mean([r.plddt for r in self.residues]))

    @property
    def high_confidence_fraction(self) -> float:
        """Fraction of residues with pLDDT > 70 (confident prediction)."""
        if not self.residues:
            return 0.0
        high = sum(1 for r in self.residues if r.plddt > 70)
        return high / len(self.residues)

    @property
    def disordered_regions(self) -> List[Tuple[int, int]]:
        """Contiguous regions with pLDDT < 50 (likely disordered)."""
        regions = []
        start = None
        for r in self.residues:
            if r.plddt < 50:
                if start is None:
                    start = r.index
            else:
                if start is not None:
                    regions.append((start, r.index - 1))
                    start = None
        if start is not None:
            regions.append((start, self.residues[-1].index))
        return regions

    def get_plddt_at(self, position: int) -> float:
        """Get pLDDT at a specific residue position (1-based)."""
        for r in self.residues:
            if r.index == position:
                return r.plddt
        return 0.0

    def get_local_plddt(self, position: int, window: int = 5) -> float:
        """Average pLDDT in a window around a position."""
        plddts = [r.plddt for r in self.residues
                  if abs(r.index - position) <= window]
        return float(np.mean(plddts)) if plddts else 0.0

    def to_dict(self) -> Dict:
        return {
            "uniprot_id": self.uniprot_id,
            "gene_name": self.gene_name,
            "organism": self.organism,
            "sequence_length": self.sequence_length,
            "mean_plddt": round(self.mean_plddt, 2),
            "high_confidence_fraction": round(self.high_confidence_fraction, 3),
            "n_disordered_regions": len(self.disordered_regions),
            "n_pockets": len(self.pockets),
            "model_version": self.model_version,
        }


# ══════════════════════════════════════════════════════════════════════
# DISEASE PROTEIN PANELS
# ══════════════════════════════════════════════════════════════════════

DISEASE_PANELS: Dict[str, Dict[str, str]] = {
    # Cancer
    "TNBC": {
        "BRCA1": "P38398", "TP53": "P04637", "EGFR": "P00533",
        "MYC": "P01106", "PTEN": "P60484", "PIK3CA": "P42336",
    },
    "GBM": {
        "IDH1": "O75874", "EGFR": "P00533", "PTEN": "P60484",
        "TP53": "P04637", "NF1": "P21359", "PDGFRA": "P16234",
    },
    "PDAC": {
        "KRAS": "P01116", "TP53": "P04637", "CDKN2A": "P42771",
        "SMAD4": "Q13315", "BRCA2": "P51587",
    },
    "AML": {
        "FLT3": "P36888", "NPM1": "P06748", "IDH1": "O75874",
        "IDH2": "P48735", "DNMT3A": "Q9Y6K1", "TP53": "P04637",
    },
    # Neurodegenerative
    "Alzheimers": {
        "APP": "P05067", "PSEN1": "P49768", "MAPT": "P10636",
        "APOE": "P02649", "BACE1": "P56817", "GSK3B": "P49841",
    },
    "Parkinsons": {
        "SNCA": "P37840", "LRRK2": "Q5S007", "PINK1": "Q9BXM7",
        "PARK7": "Q99497", "GBA": "P04062", "VPS35": "Q96QK1",
    },
    "ALS": {
        "SOD1": "P00441", "TARDBP": "Q13148", "FUS": "P35637",
        "C9orf72": "Q96LT7", "OPTN": "Q96CV9", "VCP": "P55072",
    },
    # Metabolic
    "Diabetes": {
        "INS": "P01308", "INSR": "P06213", "GCK": "P35557",
        "HNF1A": "P20823", "PPARG": "P37231", "KCNJ11": "Q14654",
    },
    # Autoimmune
    "Lupus": {
        "TREX1": "Q9NSU2", "DNASE1L3": "Q13609", "IFIH1": "Q9BYX4",
        "BLK": "P51451", "STAT4": "Q14765", "IRF5": "Q13568",
    },
}

# Map gene names to their key functional residues (active sites, catalytic residues)
GENE_KEY_RESIDUES: Dict[str, List[int]] = {
    "BRCA1": [61, 64, 1699, 1775],        # RING finger + BRCT domains
    "TP53": [175, 245, 248, 249, 273, 282],  # DNA-binding hotspots
    "EGFR": [719, 790, 858],               # Kinase domain (TKI binding)
    "IDH1": [100, 132, 170],               # Active site (2-HG production)
    "PTEN": [124, 130, 173],               # Phosphatase active site
    "SOD1": [4, 46, 93, 144],              # Cu/Zn binding + aggregation sites
    "SNCA": [30, 46, 87, 129],             # NAC region + phosphorylation
    "BACE1": [93, 289],                    # Catalytic aspartates
    "GCK": [147, 150, 228, 256],           # Glucose binding site
    "KRAS": [12, 13, 61],                  # GTPase switch regions
    "APP": [670, 671, 717],                # Secretase cleavage sites
    "LRRK2": [1441, 2019, 2020],           # ROC-COR + kinase domain
    "GPX4": [46, 73, 107],                 # Selenocysteine active site
    "HK2": [235, 466, 603],               # Glucose binding + catalytic
    "GLS": [249, 321, 394],               # Glutaminase active site
}


# ══════════════════════════════════════════════════════════════════════
# ALPHAFOLD DB REST CLIENT
# ══════════════════════════════════════════════════════════════════════

class AlphaFoldClient:
    """
    Client for the AlphaFold Protein Structure Database REST API.

    Fetches predicted structures, extracts per-residue confidence (pLDDT),
    and identifies potential binding pockets via geometric analysis.

    Usage:
        client = AlphaFoldClient()
        structure = client.fetch_structure("P04637")  # TP53
        print(structure.mean_plddt)
        print(structure.pockets)
    """

    BASE_URL = "https://alphafold.ebi.ac.uk/api"
    FILES_URL = "https://alphafold.ebi.ac.uk/files"
    CACHE_DIR = "data/alphafold_cache"

    def __init__(self, cache_dir: Optional[str] = None, use_cache: bool = True):
        self.cache_dir = Path(cache_dir or self.CACHE_DIR)
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_structure(
        self,
        uniprot_id: str,
        gene_name: str = "",
        organism: str = "Homo sapiens",
    ) -> StructureData:
        """
        Fetch and parse an AlphaFold structure for a UniProt accession.

        Tries cache first, then fetches from API. Extracts residue-level
        coordinates and pLDDT scores, then runs pocket detection.

        Args:
            uniprot_id: UniProt accession (e.g. "P04637" for TP53)
            gene_name: Gene symbol (for metadata)
            organism: Organism name (default: human)

        Returns:
            StructureData with residues, pLDDT scores, and binding pockets
        """
        # Check cache
        cache_key = f"af_{uniprot_id}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"Cache hit for {uniprot_id}")
            return cached

        # Fetch from API
        if not HAS_REQUESTS:
            raise ImportError("requests library required: pip install requests")

        cif_data = self._fetch_cif(uniprot_id)
        structure = self._parse_cif(cif_data, uniprot_id, gene_name, organism)

        # Detect binding pockets
        structure.pockets = detect_binding_pockets(structure)

        # Cache
        self._save_cache(cache_key, structure, cif_data)
        return structure

    def fetch_panel(
        self,
        disease: str,
    ) -> Dict[str, StructureData]:
        """
        Fetch all structures for a disease panel.

        Args:
            disease: Disease name (key in DISEASE_PANELS)

        Returns:
            Dict mapping gene name → StructureData
        """
        if disease not in DISEASE_PANELS:
            available = ", ".join(DISEASE_PANELS.keys())
            raise ValueError(f"Unknown disease '{disease}'. Available: {available}")

        panel = DISEASE_PANELS[disease]
        results = {}
        for gene, uniprot_id in panel.items():
            try:
                structure = self.fetch_structure(uniprot_id, gene_name=gene)
                results[gene] = structure
                logger.info(f"  {gene} ({uniprot_id}): {structure.sequence_length} residues, "
                           f"pLDDT={structure.mean_plddt:.1f}")
            except Exception as e:
                logger.warning(f"  {gene} ({uniprot_id}): FAILED — {e}")

        return results

    def _fetch_cif(self, uniprot_id: str) -> str:
        """Download mmCIF file from AlphaFold DB."""
        # Try the latest model (v4)
        url = f"{self.FILES_URL}/AF-{uniprot_id}-F1-model_v4.cif"
        logger.info(f"Fetching {url}")

        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            # Fallback to v3
            url = f"{self.FILES_URL}/AF-{uniprot_id}-F1-model_v3.cif"
            resp = requests.get(url, timeout=30)

        resp.raise_for_status()
        return resp.text

    def _parse_cif(
        self,
        cif_data: str,
        uniprot_id: str,
        gene_name: str,
        organism: str,
    ) -> StructureData:
        """
        Parse mmCIF data to extract residues with coordinates and pLDDT.

        Falls back to a lightweight regex parser if BioPython is unavailable.
        """
        if HAS_BIOPYTHON:
            return self._parse_with_biopython(cif_data, uniprot_id, gene_name, organism)
        else:
            return self._parse_lightweight(cif_data, uniprot_id, gene_name, organism)

    def _parse_with_biopython(
        self,
        cif_data: str,
        uniprot_id: str,
        gene_name: str,
        organism: str,
    ) -> StructureData:
        """Parse mmCIF using BioPython's MMCIFParser."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
            f.write(cif_data)
            tmp_path = f.name

        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(uniprot_id, tmp_path)

            residues = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] != ' ':  # Skip hetero atoms
                            continue
                        # Get Cα atom
                        if 'CA' not in residue:
                            continue
                        ca = residue['CA']
                        # pLDDT is stored in B-factor field for AlphaFold models
                        plddt = float(ca.get_bfactor())
                        coords = ca.get_vector().get_array()

                        residues.append(ResidueInfo(
                            index=residue.id[1],
                            name=residue.get_resname(),
                            plddt=plddt,
                            coords=np.array(coords, dtype=np.float64),
                        ))
                break  # Only first model

            # Estimate secondary structure from Cα geometry
            _estimate_secondary_structure(residues)

            return StructureData(
                uniprot_id=uniprot_id,
                gene_name=gene_name,
                organism=organism,
                residues=residues,
                source_url=f"{self.FILES_URL}/AF-{uniprot_id}-F1-model_v4.cif",
            )
        finally:
            os.unlink(tmp_path)

    def _parse_lightweight(
        self,
        cif_data: str,
        uniprot_id: str,
        gene_name: str,
        organism: str,
    ) -> StructureData:
        """
        Lightweight mmCIF parser — no BioPython dependency.

        Extracts Cα atoms from _atom_site loop. Handles AlphaFold's
        standard mmCIF format where pLDDT is in the B_iso_or_equiv field.
        """
        residues = []
        in_atom_site = False
        columns = []

        for line in cif_data.split('\n'):
            line = line.strip()

            if line.startswith('_atom_site.'):
                in_atom_site = True
                col_name = line.split('.')[1].strip()
                columns.append(col_name)
                continue

            if in_atom_site and line.startswith('_') and not line.startswith('_atom_site.'):
                in_atom_site = False
                continue

            if in_atom_site and line and not line.startswith('#') and not line.startswith('_'):
                parts = line.split()
                if len(parts) < len(columns):
                    continue

                col_map = {c: parts[i] for i, c in enumerate(columns) if i < len(parts)}

                atom_name = col_map.get('label_atom_id', '')
                if atom_name != 'CA':
                    continue

                try:
                    res_idx = int(col_map.get('label_seq_id', '0'))
                    res_name = col_map.get('label_comp_id', 'UNK')
                    x = float(col_map.get('Cartn_x', '0'))
                    y = float(col_map.get('Cartn_y', '0'))
                    z = float(col_map.get('Cartn_z', '0'))
                    plddt = float(col_map.get('B_iso_or_equiv', '0'))

                    residues.append(ResidueInfo(
                        index=res_idx,
                        name=res_name,
                        plddt=plddt,
                        coords=np.array([x, y, z], dtype=np.float64),
                    ))
                except (ValueError, KeyError):
                    continue

        _estimate_secondary_structure(residues)

        return StructureData(
            uniprot_id=uniprot_id,
            gene_name=gene_name,
            organism=organism,
            residues=residues,
            source_url=f"{self.FILES_URL}/AF-{uniprot_id}-F1-model_v4.cif",
        )

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cif_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.cif"

    def _load_cache(self, key: str) -> Optional[StructureData]:
        """Load a cached structure if available."""
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            residues = [
                ResidueInfo(
                    index=r['index'],
                    name=r['name'],
                    plddt=r['plddt'],
                    coords=np.array(r['coords']),
                )
                for r in data.get('residues', [])
            ]
            _estimate_secondary_structure(residues)

            pockets = [
                BindingPocket(
                    center=np.array(p['center']),
                    residue_indices=p['residue_indices'],
                    volume=p['volume'],
                    avg_plddt=p['avg_plddt'],
                    depth=p['depth'],
                    druggability_score=p['druggability_score'],
                )
                for p in data.get('pockets', [])
            ]

            return StructureData(
                uniprot_id=data['uniprot_id'],
                gene_name=data.get('gene_name', ''),
                organism=data.get('organism', ''),
                residues=residues,
                pockets=pockets,
                model_version=data.get('model_version', ''),
                source_url=data.get('source_url', ''),
            )
        except Exception as e:
            logger.warning(f"Cache read failed for {key}: {e}")
            return None

    def _save_cache(self, key: str, structure: StructureData, cif_data: str = ""):
        """Save structure to cache."""
        if not self.use_cache:
            return
        data = {
            'uniprot_id': structure.uniprot_id,
            'gene_name': structure.gene_name,
            'organism': structure.organism,
            'model_version': structure.model_version,
            'source_url': structure.source_url,
            'residues': [
                {
                    'index': r.index,
                    'name': r.name,
                    'plddt': r.plddt,
                    'coords': r.coords.tolist(),
                }
                for r in structure.residues
            ],
            'pockets': [
                {
                    'center': p.center.tolist(),
                    'residue_indices': p.residue_indices,
                    'volume': p.volume,
                    'avg_plddt': p.avg_plddt,
                    'depth': p.depth,
                    'druggability_score': p.druggability_score,
                }
                for p in structure.pockets
            ],
        }
        with open(self._cache_path(key), 'w') as f:
            json.dump(data, f, indent=2)
        if cif_data:
            with open(self._cif_cache_path(key), 'w') as f:
                f.write(cif_data)


# ══════════════════════════════════════════════════════════════════════
# GEOMETRIC POCKET DETECTION
# ══════════════════════════════════════════════════════════════════════

def _estimate_secondary_structure(residues: List[ResidueInfo]):
    """
    Estimate secondary structure from Cα distances.

    Heuristic based on Cα(i)-Cα(i+3) distance:
      - Helix: ~5.0-5.5 Å
      - Sheet: ~9-11 Å (extended)
      - Coil: everything else
    """
    if len(residues) < 4:
        return

    for i in range(len(residues) - 3):
        d = np.linalg.norm(residues[i].coords - residues[i + 3].coords)
        if 4.5 <= d <= 6.0:
            residues[i].secondary_structure = "H"
            residues[i + 1].secondary_structure = "H"
            residues[i + 2].secondary_structure = "H"
            residues[i + 3].secondary_structure = "H"
        elif d > 9.0:
            if residues[i].secondary_structure == "C":
                residues[i].secondary_structure = "E"
            if residues[i + 3].secondary_structure == "C":
                residues[i + 3].secondary_structure = "E"


def detect_binding_pockets(
    structure: StructureData,
    probe_radius: float = 1.4,
    min_pocket_residues: int = 8,
    max_pockets: int = 5,
) -> List[BindingPocket]:
    """
    Detect binding pockets via geometric analysis of Cα coordinates.

    Algorithm:
      1. Build a distance matrix of all Cα atoms
      2. Identify concave regions (residues buried below the surface)
      3. Cluster buried residues by spatial proximity
      4. Score clusters for druggability (size, depth, confidence)

    This is a simplified approach suitable for rapid screening.
    For high-precision docking, use dedicated tools (FPOCKET, SiteMap).
    """
    if len(structure.residues) < min_pocket_residues:
        return []

    coords = np.array([r.coords for r in structure.residues])
    n = len(coords)

    # Step 1: Compute centroid and per-residue distance to centroid
    centroid = np.mean(coords, axis=0)
    dist_to_centroid = np.linalg.norm(coords - centroid, axis=1)

    # Step 2: Identify concavity via local buriedness
    # A residue is "buried" if its average distance to neighbors is
    # small compared to surface residues
    buriedness = np.zeros(n)
    neighbor_cutoff = 10.0  # Å

    for i in range(n):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        neighbors = dists < neighbor_cutoff
        neighbor_count = np.sum(neighbors) - 1  # exclude self
        if neighbor_count > 0:
            avg_dist = np.mean(dists[neighbors & (dists > 0)])
            buriedness[i] = neighbor_count / (avg_dist + 1e-10)

    # Normalize buriedness to 0-1
    if np.max(buriedness) > 0:
        buriedness /= np.max(buriedness)

    # Step 3: Find concave clusters (buried residues near each other)
    buried_threshold = 0.4
    buried_indices = np.where(buriedness > buried_threshold)[0]

    if len(buried_indices) < min_pocket_residues:
        # Lower threshold
        buried_threshold = 0.25
        buried_indices = np.where(buriedness > buried_threshold)[0]

    if len(buried_indices) < min_pocket_residues:
        return []

    # Simple agglomerative clustering
    clusters = _cluster_residues(coords[buried_indices], buried_indices,
                                  cutoff=8.0, min_size=min_pocket_residues)

    # Step 4: Score and rank pockets
    pockets = []
    for cluster_indices in clusters[:max_pockets]:
        pocket_coords = coords[cluster_indices]
        pocket_center = np.mean(pocket_coords, axis=0)

        # Volume estimate (convex hull approximation)
        ranges = np.ptp(pocket_coords, axis=0)
        volume = float(np.prod(ranges) * 0.5236)  # Ellipsoid approximation

        # Depth: max distance from pocket center to any pocket residue
        depth = float(np.max(np.linalg.norm(pocket_coords - pocket_center, axis=1)))

        # Average pLDDT of pocket residues
        pocket_plddts = [structure.residues[i].plddt for i in cluster_indices
                         if i < len(structure.residues)]
        avg_plddt = float(np.mean(pocket_plddts)) if pocket_plddts else 0.0

        # Druggability heuristic
        # High confidence + good volume + moderate depth = druggable
        vol_score = min(1.0, volume / 1000.0)       # Sweet spot: 300-800 ų
        depth_score = min(1.0, depth / 12.0)         # Deeper pockets are better
        conf_score = avg_plddt / 100.0               # Higher pLDDT = more reliable
        size_score = min(1.0, len(cluster_indices) / 20.0)
        druggability = 0.3 * vol_score + 0.25 * depth_score + 0.25 * conf_score + 0.2 * size_score

        pockets.append(BindingPocket(
            center=pocket_center,
            residue_indices=sorted([int(i) for i in cluster_indices]),
            volume=round(volume, 1),
            avg_plddt=round(avg_plddt, 1),
            depth=round(depth, 1),
            druggability_score=round(druggability, 3),
        ))

    # Sort by druggability (best first)
    pockets.sort(key=lambda p: p.druggability_score, reverse=True)
    return pockets


def _cluster_residues(
    coords: np.ndarray,
    original_indices: np.ndarray,
    cutoff: float = 8.0,
    min_size: int = 5,
) -> List[np.ndarray]:
    """
    Simple single-linkage clustering of residue coordinates.

    Returns list of arrays, each containing original indices for a cluster.
    """
    n = len(coords)
    assigned = np.full(n, -1, dtype=int)
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0:
            continue

        # BFS from this point
        queue = [i]
        members = []
        while queue:
            current = queue.pop(0)
            if assigned[current] >= 0:
                continue
            assigned[current] = cluster_id
            members.append(current)

            # Find unassigned neighbors within cutoff
            dists = np.linalg.norm(coords - coords[current], axis=1)
            for j in range(n):
                if assigned[j] < 0 and dists[j] < cutoff:
                    queue.append(j)

        cluster_id += 1

    # Gather clusters
    clusters = []
    for cid in range(cluster_id):
        members = np.where(assigned == cid)[0]
        if len(members) >= min_size:
            clusters.append(original_indices[members])

    # Sort by size (largest first)
    clusters.sort(key=len, reverse=True)
    return clusters


# ══════════════════════════════════════════════════════════════════════
# MOCK DATA (for testing without API access)
# ══════════════════════════════════════════════════════════════════════

def create_mock_structure(
    uniprot_id: str = "P04637",
    gene_name: str = "TP53",
    n_residues: int = 393,
    seed: int = 42,
) -> StructureData:
    """
    Create a realistic mock AlphaFold structure for testing.

    Generates synthetic coordinates and pLDDT profiles that mimic
    real AlphaFold predictions (ordered domains + disordered tails).
    """
    rng = np.random.RandomState(seed)

    residues = []
    # Generate a folded protein-like structure
    # Use a helix-turn-helix pattern with disordered termini
    for i in range(1, n_residues + 1):
        # Position along a coiled path
        t = i / n_residues
        theta = t * 20 * np.pi  # ~10 turns
        r = 15.0 + 5.0 * np.sin(t * 4 * np.pi)  # Varying radius

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = t * 50.0 + 2.0 * np.sin(theta * 0.5)

        # Add noise
        coords = np.array([x, y, z]) + rng.randn(3) * 0.5

        # pLDDT profile: high in core, low at termini
        if t < 0.05 or t > 0.95:
            plddt = rng.uniform(30, 50)  # Disordered termini
        elif 0.3 < t < 0.7:
            plddt = rng.uniform(80, 95)  # Well-folded core
        else:
            plddt = rng.uniform(60, 85)  # Moderate confidence

        aa_names = ['ALA', 'GLY', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE',
                    'TRP', 'MET', 'SER', 'THR', 'CYS', 'TYR', 'ASN',
                    'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS']

        residues.append(ResidueInfo(
            index=i,
            name=rng.choice(aa_names),
            plddt=round(plddt, 1),
            coords=coords,
        ))

    _estimate_secondary_structure(residues)

    structure = StructureData(
        uniprot_id=uniprot_id,
        gene_name=gene_name,
        organism="Homo sapiens",
        residues=residues,
        model_version="mock_v1",
        source_url="mock://test",
    )
    structure.pockets = detect_binding_pockets(structure)
    return structure

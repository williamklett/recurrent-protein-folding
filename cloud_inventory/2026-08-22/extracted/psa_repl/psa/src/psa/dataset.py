"""Hand-curated starter set of small, high-resolution PDB monomers spanning
fold classes (all-α, all-β, α+β, α/β). The list is intentionally biased toward
classic, well-characterized proteins so that any debugging in early blocks can
be compared to known biology.

Each entry: (pdb_id, chain_id, description). chain_id may be None to take the
first protein chain found.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetProtein:
    pdb_id: str
    chain_id: str | None
    fold_class: str  # "alpha", "beta", "alpha_beta", "mixed"
    description: str


STARTER_SET: list[TargetProtein] = [
    # --- all-α ---
    TargetProtein("1VII", "A", "alpha", "villin headpiece HP35"),
    TargetProtein("1L2Y", "A", "alpha", "Trp-cage miniprotein"),
    TargetProtein("1ENH", "A", "alpha", "engrailed homeodomain"),
    TargetProtein("2P6J", "A", "alpha", "designed homeodomain"),
    TargetProtein("1HRC", "A", "alpha", "horse heart cytochrome c"),
    TargetProtein("1BDD", "A", "alpha", "protein A B-domain"),
    TargetProtein("1LMB", "3", "alpha", "lambda repressor N-terminal"),
    TargetProtein("1HZ6", "A", "alpha", "three-helix bundle"),
    TargetProtein("2A3D", "A", "alpha", "designed three-helix bundle"),
    TargetProtein("1ROP", "A", "alpha", "ColE1 Rop (four-helix bundle)"),
    # --- all-β ---
    TargetProtein("1PIN", "A", "beta", "Pin1 WW domain"),
    TargetProtein("1SHG", "A", "beta", "α-spectrin SH3"),
    TargetProtein("1MJC", "A", "beta", "E. coli cold shock CspA"),
    TargetProtein("1FYN", "A", "beta", "Fyn SH3"),
    TargetProtein("1UAO", "A", "beta", "chignolin β-hairpin"),
    TargetProtein("2EZK", "A", "beta", "small β-sandwich"),
    TargetProtein("1AYE", "A", "beta", "PDZ domain"),
    TargetProtein("1QYS", "A", "beta", "Top7 β-sheet (designed)"),
    TargetProtein("2PTL", "A", "beta", "protein L β-grasp"),
    TargetProtein("1WHZ", "A", "beta", "monobody β-sandwich"),
    # --- α+β / α/β ---
    TargetProtein("1UBQ", "A", "alpha_beta", "ubiquitin"),
    TargetProtein("2GB1", "A", "alpha_beta", "protein G B1 (NMR)"),
    TargetProtein("2QMT", "A", "alpha_beta", "protein G B1 (crystal)"),
    TargetProtein("1CRN", "A", "alpha_beta", "crambin (plant defensin)"),
    TargetProtein("1IGD", "A", "alpha_beta", "protein G B3"),
    TargetProtein("1BPI", "A", "alpha_beta", "BPTI"),
    TargetProtein("5PTI", "A", "alpha_beta", "BPTI (different solve)"),
    TargetProtein("1B9C", "A", "alpha_beta", "small kinase-like"),
    TargetProtein("1AIL", "A", "alpha_beta", "interferon α2 fragment"),
    TargetProtein("1NXB", "A", "alpha_beta", "neurotoxin B"),
    TargetProtein("1KNT", "A", "alpha_beta", "Kunitz domain"),
    TargetProtein("3CHY", "A", "alpha_beta", "CheY response regulator"),
    TargetProtein("7RSA", "A", "alpha_beta", "ribonuclease A"),
    TargetProtein("1LYZ", "A", "alpha_beta", "hen egg-white lysozyme"),
    TargetProtein("1HEW", "A", "alpha_beta", "lysozyme (alt soak)"),
    TargetProtein("2CI2", "I", "alpha_beta", "chymotrypsin inhibitor 2"),
    TargetProtein("1POH", "A", "alpha_beta", "histidine-containing protein HPr"),
    TargetProtein("1AKI", "A", "alpha_beta", "lysozyme (alt)"),
    TargetProtein("1RNB", "A", "alpha_beta", "barnase"),
    TargetProtein("1A6N", "A", "alpha_beta", "myoglobin-like"),
    # --- mixed / β-rich / unique folds ---
    TargetProtein("1ACF", "A", "mixed", "actin-binding fragment"),
    TargetProtein("1AZI", "A", "mixed", "azurin (cupredoxin)"),
    TargetProtein("1B3A", "A", "mixed", "ferredoxin-like"),
    TargetProtein("1CSP", "A", "mixed", "B. subtilis cold shock CspB"),
    TargetProtein("1FNF", "A", "mixed", "fibronectin type-III"),
    TargetProtein("1NYF", "A", "mixed", "Nck SH3"),
    TargetProtein("1PGB", "A", "mixed", "protein G B1 (orig)"),
    TargetProtein("1TIT", "A", "mixed", "titin Ig"),
    TargetProtein("1WIT", "A", "mixed", "twitchin Ig"),
    TargetProtein("2DRI", "A", "mixed", "D-ribose binding (large; may exclude)"),
    TargetProtein("1KAO", "A", "mixed", "domain swap"),
    TargetProtein("1AGY", "A", "mixed", "cutinase"),
    TargetProtein("1IFC", "A", "mixed", "fatty acid binding"),
    TargetProtein("1KTE", "A", "mixed", "thioredoxin-like"),
    TargetProtein("1A2P", "A", "mixed", "barnase trimer monomer"),
    TargetProtein("1CTF", "A", "mixed", "L7/L12 50S ribosomal C-term"),
    TargetProtein("1FAS", "A", "mixed", "fasciculin"),
    TargetProtein("1GB1", "A", "mixed", "protein G B1 (alt)"),
    TargetProtein("1HHP", "A", "mixed", "HIV-1 protease monomer"),
    TargetProtein("1MBA", "A", "mixed", "aplysia myoglobin"),
]


def pdb_url(pdb_id: str) -> str:
    """Standard RCSB mmCIF URL."""
    return f"https://files.rcsb.org/download/{pdb_id.lower()}.cif"

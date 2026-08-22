"""Compute physical quantities from a protein structure file (PDB/mmCIF).

These quantities form the bottleneck vocabulary z_phys of the PSA.

For a structure with L residues, we return:

  Pair (L, L):
    distances     Cα–Cα Euclidean distance (Å)
    contacts      d_ij < 8 Å
    hbonds        Baker-Hubbard hydrogen bonds
    seq_sep       |i − j| (integer, also stored log-scaled)
    orient_cos    cos angle between Cβ vectors at i and j

  Single (L,):
    ss            DSSP 8-class secondary structure (int 0-7)
    phi / psi / omega  backbone dihedrals (sin, cos pairs)
    sasa          solvent-accessible surface area (Å²)
    burial        number of Cα within 10 Å (excluding self)
    sequence      one-letter amino-acid sequence
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import mdtraj as md
import numpy as np

log = logging.getLogger(__name__)

# DSSP 8-class label vocabulary. mdtraj returns single-letter codes; we map to ints
# so downstream code can use them as embedding indices.
# 'NA' is used for residues missing structure annotation; treated as its own class.
DSSP_LABELS = ["H", "B", "E", "G", "I", "T", "S", " ", "NA"]
DSSP_LABEL_TO_INDEX = {label: i for i, label in enumerate(DSSP_LABELS)}


@dataclasses.dataclass
class Physics:
    """Container for all physical quantities computed for one chain."""

    sequence: str
    # Pair features, shape (L, L)
    distances: np.ndarray
    contacts: np.ndarray
    hbonds: np.ndarray
    seq_sep: np.ndarray
    orient_cos: np.ndarray
    # Single features, shape (L,) or (L, 2) for angles
    ss: np.ndarray
    phi: np.ndarray  # (L, 2): sin, cos
    psi: np.ndarray  # (L, 2)
    omega: np.ndarray  # (L, 2)
    sasa: np.ndarray
    burial: np.ndarray
    # K=2-derived per-residue aggregates (continuous geometry summaries).
    # Defined so an SAE feature can be regressed against them to test whether
    # it encodes a continuous local-geometry property rather than a discrete one.
    mean_dist_to_5nn: np.ndarray              # mean Cα distance to 5 nearest residues (Å)
    n_contacts: np.ndarray                    # # of contact partners (d < 8 Å)
    mean_orient_cos_with_contacts: np.ndarray  # mean Cβ·Cβ over contact partners (0 if none)
    orient_variance_with_neighbors: np.ndarray  # var(orient_cos) over neighbors within 10 Å

    @property
    def L(self) -> int:
        return len(self.sequence)

    def to_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            sequence=np.array(self.sequence),
            distances=self.distances.astype(np.float32),
            contacts=self.contacts.astype(np.bool_),
            hbonds=self.hbonds.astype(np.bool_),
            seq_sep=self.seq_sep.astype(np.int16),
            orient_cos=self.orient_cos.astype(np.float32),
            ss=self.ss.astype(np.int8),
            phi=self.phi.astype(np.float32),
            psi=self.psi.astype(np.float32),
            omega=self.omega.astype(np.float32),
            sasa=self.sasa.astype(np.float32),
            burial=self.burial.astype(np.float32),
            mean_dist_to_5nn=self.mean_dist_to_5nn.astype(np.float32),
            n_contacts=self.n_contacts.astype(np.int16),
            mean_orient_cos_with_contacts=self.mean_orient_cos_with_contacts.astype(np.float32),
            orient_variance_with_neighbors=self.orient_variance_with_neighbors.astype(np.float32),
        )


def _load_first_protein_chain(structure_path: str | Path) -> md.Trajectory:
    """Load structure and restrict to the first protein chain.

    Drops waters, ions, ligands, and any chain other than the first protein chain.
    """
    traj = md.load(str(structure_path))
    top = traj.topology

    # Find the first chain that has any standard amino-acid residue.
    first_chain_index = None
    for chain in top.chains:
        if any(r.is_protein for r in chain.residues):
            first_chain_index = chain.index
            break
    if first_chain_index is None:
        raise ValueError(f"No protein chain found in {structure_path}")

    keep_atoms = top.select(f"protein and chainid {first_chain_index}")
    if len(keep_atoms) == 0:
        raise ValueError(f"Empty protein selection for {structure_path}")
    return traj.atom_slice(keep_atoms)


def _residue_alpha_carbons(traj: md.Trajectory) -> tuple[np.ndarray, list[md.core.topology.Residue]]:
    """Return (ca_indices, residues) for residues that have a Cα atom.

    Residues without a Cα (rare gaps, non-standard) are dropped from the L-axis.
    """
    residues = []
    ca_indices = []
    for residue in traj.topology.residues:
        ca = next((a for a in residue.atoms if a.name == "CA"), None)
        if ca is None:
            log.warning("Residue %s has no Cα; skipping", residue)
            continue
        residues.append(residue)
        ca_indices.append(ca.index)
    return np.array(ca_indices, dtype=np.int64), residues


def _compute_cb_vectors(traj: md.Trajectory, residues) -> np.ndarray:
    """Return (L, 3) array of unit Cβ vectors (Cα→Cβ).

    For glycine (no Cβ) we synthesize a pseudo-Cβ from N, Cα, C using the standard
    tetrahedral geometry: pseudo_Cβ ≈ Cα + (Cα−mid(N,C)) rotated; here we use the
    simpler heuristic of placing it opposite the centroid of the backbone neighbors.
    """
    xyz = traj.xyz[0]  # (atoms, 3), in nm
    cb_vecs = np.zeros((len(residues), 3), dtype=np.float32)
    for i, residue in enumerate(residues):
        atoms = {a.name: a.index for a in residue.atoms}
        if "CA" not in atoms:
            continue
        ca = xyz[atoms["CA"]]
        cb_idx = atoms.get("CB")
        if cb_idx is not None:
            v = xyz[cb_idx] - ca
        else:
            n = xyz[atoms.get("N", atoms["CA"])]
            c = xyz[atoms.get("C", atoms["CA"])]
            # Pseudo-Cβ direction: away from centroid of N, C
            v = ca - 0.5 * (n + c)
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            cb_vecs[i] = (v / norm).astype(np.float32)
    return cb_vecs


def _pad_to_residue_axis(values: np.ndarray, target_L: int, fill: float) -> np.ndarray:
    """Pad a (K,) or (K, D) array to length target_L along axis 0 with `fill`."""
    if values.shape[0] == target_L:
        return values
    pad_shape = (target_L - values.shape[0],) + values.shape[1:]
    pad = np.full(pad_shape, fill, dtype=values.dtype)
    return np.concatenate([values, pad], axis=0)


def compute_physics(structure_path: str | Path) -> Physics:
    """Compute the full physical-quantity dict for one structure.

    Distances are returned in Ångström. mdtraj works in nanometres internally so
    we multiply Cα distance arrays by 10.
    """
    traj = _load_first_protein_chain(structure_path)
    ca_indices, residues = _residue_alpha_carbons(traj)
    L = len(residues)
    if L < 5:
        raise ValueError(f"{structure_path} has only {L} residues; need ≥ 5")

    seq = "".join(r.code if r.code else "X" for r in residues)

    # --- Pair: Cα–Cα distance and contact ---
    ca_xyz_nm = traj.xyz[0, ca_indices]  # (L, 3) in nm
    diff = ca_xyz_nm[:, None, :] - ca_xyz_nm[None, :, :]
    dist_A = np.linalg.norm(diff, axis=-1) * 10.0  # Å
    contacts = (dist_A < 8.0) & (dist_A > 0.0)

    # --- Pair: hydrogen bonds (Kabsch–Sander / DSSP energy criterion) ---
    # Works on heavy-atom-only structures; mdtraj synthesizes virtual hydrogens
    # internally for the backbone N–H. Returns a list of (n_pairs, 2) arrays of
    # (donor_residue_index, acceptor_residue_index) per frame.
    residue_index_to_position = {r.index: pos for pos, r in enumerate(residues)}
    hbonds = np.zeros((L, L), dtype=bool)
    try:
        ks_bonds = md.kabsch_sander(traj)[0]  # csr_matrix (n_residues × n_residues), energy values
        ks_dense = ks_bonds.toarray() if hasattr(ks_bonds, "toarray") else np.asarray(ks_bonds)
        # Nonzero entries = detected H-bond (negative energy).
        donor_idx, acceptor_idx = np.nonzero(ks_dense)
        for d, a in zip(donor_idx, acceptor_idx):
            ri = residue_index_to_position.get(int(d))
            rj = residue_index_to_position.get(int(a))
            if ri is None or rj is None:
                continue
            hbonds[ri, rj] = True
            hbonds[rj, ri] = True  # symmetrize — direction recorded only as presence
    except Exception as e:
        log.warning("kabsch_sander failed for %s: %s; hbonds left empty", structure_path, e)

    # --- Pair: sequence separation ---
    idx = np.arange(L)
    seq_sep = np.abs(idx[:, None] - idx[None, :]).astype(np.int16)

    # --- Pair: Cβ orientation cosine ---
    cb_vecs = _compute_cb_vectors(traj, residues)  # (L, 3)
    orient_cos = (cb_vecs @ cb_vecs.T).astype(np.float32)

    # --- Single: DSSP ---
    # mdtraj returns shape (n_frames, n_residues_with_dssp) of one-letter codes.
    try:
        dssp = md.compute_dssp(traj, simplified=False)[0]  # (n_residues,)
    except Exception as e:
        log.warning("DSSP failed for %s: %s; using NA for all residues", structure_path, e)
        dssp = np.array(["NA"] * L)
    # mdtraj may return a different length than L if it dropped non-standard residues;
    # pad to L with NA.
    if len(dssp) < L:
        dssp = np.concatenate([dssp, np.array(["NA"] * (L - len(dssp)))])
    ss = np.array(
        [DSSP_LABEL_TO_INDEX.get(c, DSSP_LABEL_TO_INDEX["NA"]) for c in dssp[:L]],
        dtype=np.int8,
    )

    # --- Single: backbone dihedrals φ ψ ω ---
    # mdtraj returns (atom_quartets, dihedral_values_per_frame). Dihedrals are
    # defined only for interior residues; we pad missing positions with 0 → sin=0, cos=1.
    def angles_to_sincos(angle_residue_indices: np.ndarray, angle_values: np.ndarray) -> np.ndarray:
        # Pad missing entries with angle = 0 → (sin, cos) = (0, 1), preserving unit norm.
        out = np.zeros((L, 2), dtype=np.float32)
        out[:, 1] = 1.0
        n = len(angle_values)
        out[:n, 0] = np.sin(angle_values).astype(np.float32)
        out[:n, 1] = np.cos(angle_values).astype(np.float32)
        return out

    phi_idx, phi_vals = md.compute_phi(traj)
    psi_idx, psi_vals = md.compute_psi(traj)
    omega_idx, omega_vals = md.compute_omega(traj)
    phi = angles_to_sincos(phi_idx, phi_vals[0])
    psi = angles_to_sincos(psi_idx, psi_vals[0])
    omega = angles_to_sincos(omega_idx, omega_vals[0])

    # --- Single: SASA (per-residue, Å²) ---
    sasa_per_atom_nm2 = md.shrake_rupley(traj, mode="residue")[0]  # (n_residues,)
    sasa = _pad_to_residue_axis(sasa_per_atom_nm2, L, fill=0.0) * 100.0  # nm² → Å²

    # --- Single: burial (count of other Cα within 10 Å) ---
    burial = ((dist_A < 10.0) & (dist_A > 0.0)).sum(axis=1).astype(np.float32)

    # --- K=2-derived per-residue aggregates ---
    # mean distance to 5 nearest neighbors (excluding self).
    # We sort each row's non-diagonal distances ascending and take the mean of the
    # first 5 (or fewer if L is small).
    off_diag_dist = dist_A.copy()
    np.fill_diagonal(off_diag_dist, np.inf)
    k_nn = min(5, L - 1)
    nearest_k = np.partition(off_diag_dist, k_nn - 1, axis=1)[:, :k_nn]
    mean_dist_to_5nn = nearest_k.mean(axis=1).astype(np.float32)

    n_contacts = contacts.sum(axis=1).astype(np.int16)

    # Mean orient_cos over contact partners; 0 where the residue has no contacts.
    contact_mask = contacts.astype(np.float32)
    contact_orient_sum = (contact_mask * orient_cos).sum(axis=1)
    mean_orient_cos_with_contacts = np.divide(
        contact_orient_sum,
        n_contacts.astype(np.float32),
        out=np.zeros(L, dtype=np.float32),
        where=n_contacts > 0,
    )

    # Variance of orient_cos over neighbors (d < 10 Å), excluding self.
    neighbor_mask = ((dist_A < 10.0) & (dist_A > 0.0)).astype(np.float32)
    n_neighbors = neighbor_mask.sum(axis=1)
    neighbor_orient_sum = (neighbor_mask * orient_cos).sum(axis=1)
    neighbor_orient_mean = np.divide(
        neighbor_orient_sum, n_neighbors,
        out=np.zeros(L, dtype=np.float32), where=n_neighbors > 0,
    )
    # E[(X - mean)^2] over the neighbor distribution
    diff_sq = (orient_cos - neighbor_orient_mean[:, None]) ** 2 * neighbor_mask
    orient_variance_with_neighbors = np.divide(
        diff_sq.sum(axis=1), n_neighbors,
        out=np.zeros(L, dtype=np.float32), where=n_neighbors > 0,
    ).astype(np.float32)

    return Physics(
        sequence=seq,
        distances=dist_A.astype(np.float32),
        contacts=contacts,
        hbonds=hbonds,
        seq_sep=seq_sep,
        orient_cos=orient_cos,
        ss=ss,
        phi=phi,
        psi=psi,
        omega=omega,
        sasa=sasa.astype(np.float32),
        burial=burial,
        mean_dist_to_5nn=mean_dist_to_5nn,
        n_contacts=n_contacts,
        mean_orient_cos_with_contacts=mean_orient_cos_with_contacts,
        orient_variance_with_neighbors=orient_variance_with_neighbors,
    )

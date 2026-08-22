from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch


AA3_TO_ID = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}


@dataclass(frozen=True)
class MdcathWindow:
    seq: torch.Tensor
    start: torch.Tensor
    target: torch.Tensor


class MdcathTrajectory:
    def __init__(
        self,
        path: Path,
        domain: str | None = None,
        temperature: str = "320",
        replica: str = "0",
        frame_stride: int = 1,
    ) -> None:
        self.path = Path(path)
        self.domain = domain
        self.temperature = temperature
        self.replica = replica
        self.frame_stride = frame_stride
        self._h5: h5py.File | None = None
        self._coords = None
        self.ca_indices: list[int] = []
        self.sequence: list[int] = []
        self.num_frames = 0
        self.length = 0
        self._open()

    def _open(self) -> None:
        self._h5 = h5py.File(self.path, "r")
        if self.domain is None:
            self.domain = next(iter(self._h5.keys()))
        domain_group = self._h5[self.domain]
        self.ca_indices, resnames = parse_ca_atoms(read_scalar_text(domain_group["pdbProteinAtoms"]))
        self.sequence = [AA3_TO_ID.get(name, 0) for name in resnames]
        self.length = len(self.sequence)
        self._coords = domain_group[self.temperature][self.replica]["coords"]
        self.num_frames = int(self._coords.shape[0])

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "MdcathTrajectory":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def sample_batch(
        self,
        batch_size: int,
        window: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> MdcathWindow:
        max_start = self.num_frames - (window * self.frame_stride) - 1
        if max_start < 1:
            raise ValueError("Trajectory is too short for requested window and stride")
        starts = rng.integers(0, max_start, size=batch_size)
        start_coords = []
        target_coords = []
        for frame in starts:
            start_coords.append(self._frame_ca(int(frame)))
            target_frames = [int(frame + (i + 1) * self.frame_stride) for i in range(window)]
            target_coords.append(np.stack([self._frame_ca(idx) for idx in target_frames], axis=0))
        seq = torch.tensor(self.sequence, dtype=torch.long, device=device)[None].repeat(batch_size, 1)
        start = torch.tensor(np.stack(start_coords, axis=0), dtype=torch.float32, device=device)
        target = torch.tensor(np.stack(target_coords, axis=0), dtype=torch.float32, device=device)
        start = start - start.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=2, keepdim=True)
        return MdcathWindow(seq=seq, start=start, target=target)

    def _frame_ca(self, frame: int) -> np.ndarray:
        return np.asarray(self._coords[frame, self.ca_indices, :], dtype=np.float32)


def read_scalar_text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_ca_atoms(pdb_text: str) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    resnames: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        serial = int(line[6:11]) - 1
        resname = line[17:20].strip()
        chain = line[21:22].strip()
        resid = line[22:26].strip()
        key = (chain, resid, resname)
        if key in seen:
            continue
        seen.add(key)
        indices.append(serial)
        resnames.append(resname)
    if not indices:
        raise ValueError("No CA atoms found in mdCATH PDB metadata")
    return indices, resnames


def available_trajectories(path: Path) -> Sequence[tuple[str, str, str, int]]:
    with h5py.File(path, "r") as h5:
        rows = []
        for domain in h5.keys():
            if not isinstance(h5[domain], h5py.Group):
                continue
            for temperature in h5[domain].keys():
                if not temperature.isdigit():
                    continue
                for replica in h5[domain][temperature].keys():
                    group = h5[domain][temperature][replica]
                    rows.append((domain, temperature, replica, int(group["coords"].shape[0])))
        return rows


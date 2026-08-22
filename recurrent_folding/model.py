from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentFolder(nn.Module):
    def __init__(
        self,
        aa_count: int = 20,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        steps: int = 8,
        max_delta: float = 1.25,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.max_delta = max_delta
        self.embed = nn.Embedding(aa_count, embed_dim)
        self.context = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.cell = nn.LSTMCell(hidden_dim + 3, hidden_dim)
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def initial_backbone(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        x = torch.arange(length, device=device, dtype=torch.float32) * 3.8
        coords = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
        coords = coords - coords.mean(dim=0, keepdim=True)
        return coords[None, :, :].repeat(batch_size, 1, 1)

    def forward(self, seq: torch.Tensor, initial_coords: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, length = seq.shape
        device = seq.device
        emb = self.embed(seq)
        context, _ = self.context(emb)
        if initial_coords is None:
            coords = self.initial_backbone(batch_size, length, device)
        else:
            coords = initial_coords - initial_coords.mean(dim=1, keepdim=True)
        h = torch.zeros(batch_size * length, context.shape[-1], device=device)
        c = torch.zeros_like(h)
        trajectory = []
        context_flat = context.reshape(batch_size * length, -1)

        for _ in range(self.steps):
            inp = torch.cat([context_flat, coords.reshape(batch_size * length, 3)], dim=-1)
            h, c = self.cell(inp, (h, c))
            delta = torch.tanh(self.delta(h)).reshape(batch_size, length, 3) * self.max_delta
            coords = coords + delta
            coords = coords - coords.mean(dim=1, keepdim=True)
            trajectory.append(coords)

        return torch.stack(trajectory, dim=1)


class PairwiseRecurrentFolder(nn.Module):
    """Two-channel pairwise recurrent dynamics.

    `latent_pair` is unconstrained bioinformatic memory. `physical_pair` is the
    only channel decoded into coordinate motion, so the model can reason in
    latent space but must express its answer as iterative physical updates.
    """

    def __init__(
        self,
        aa_count: int = 20,
        embed_dim: int = 48,
        pair_dim: int = 96,
        physical_dim: int | None = None,
        steps: int = 8,
        max_delta: float = 0.8,
        max_sep: float = 32.0,
        bond_length: float = 3.8,
        bond_project_iters: int = 2,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.max_delta = max_delta
        self.max_sep = max_sep
        self.bond_length = bond_length
        self.bond_project_iters = bond_project_iters
        self.physical_dim = physical_dim or max(pair_dim // 2, 16)
        self.embed = nn.Embedding(aa_count, embed_dim)
        pair_input_dim = embed_dim * 2 + 2
        geom_dim = 9
        self.latent_init = nn.Sequential(
            nn.Linear(pair_input_dim, pair_dim),
            nn.SiLU(),
            nn.LayerNorm(pair_dim),
        )
        self.physical_init = nn.Sequential(
            nn.Linear(geom_dim, self.physical_dim),
            nn.SiLU(),
            nn.LayerNorm(self.physical_dim),
        )
        coupled_dim = pair_dim + self.physical_dim + geom_dim
        self.latent_cell = nn.LSTMCell(coupled_dim, pair_dim)
        self.physical_cell = nn.LSTMCell(coupled_dim, self.physical_dim)
        self.physical_head = nn.Sequential(
            nn.LayerNorm(self.physical_dim + geom_dim),
            nn.Linear(self.physical_dim + geom_dim, self.physical_dim),
            nn.SiLU(),
            nn.Linear(self.physical_dim, 1),
        )
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(pair_dim + self.physical_dim + geom_dim),
            nn.Linear(pair_dim + self.physical_dim + geom_dim, pair_dim),
            nn.SiLU(),
            nn.Linear(pair_dim, 38),
        )
        nn.init.zeros_(self.physical_head[-1].weight)
        nn.init.zeros_(self.physical_head[-1].bias)

    def initial_backbone(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        x = torch.arange(length, device=device, dtype=torch.float32) * 3.8
        coords = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
        coords = coords - coords.mean(dim=0, keepdim=True)
        return coords[None, :, :].repeat(batch_size, 1, 1)

    def forward(
        self,
        seq: torch.Tensor,
        initial_coords: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, length = seq.shape
        device = seq.device
        coords = (
            self.initial_backbone(batch_size, length, device)
            if initial_coords is None
            else initial_coords - initial_coords.mean(dim=1, keepdim=True)
        )
        emb = self.embed(seq)
        geom = self._geometry_features(coords)
        latent_pair = self._initial_latent_pair(emb)
        physical_pair = self.physical_init(geom)
        latent_h = latent_pair.reshape(batch_size * length * length, -1)
        latent_c = torch.zeros_like(latent_h)
        physical_h = physical_pair.reshape(batch_size * length * length, -1)
        physical_c = torch.zeros_like(physical_h)
        trajectory = []

        eye = torch.eye(length, device=device, dtype=torch.bool)[None, :, :]
        offdiag = (~eye).float()

        for _ in range(self.steps):
            geom = self._geometry_features(coords)
            coupled = torch.cat([latent_pair, physical_pair, geom], dim=-1)
            cell_in = coupled.reshape(batch_size * length * length, -1)
            latent_h, latent_c = self.latent_cell(cell_in, (latent_h, latent_c))
            physical_h, physical_c = self.physical_cell(cell_in, (physical_h, physical_c))
            latent_pair = latent_h.reshape(batch_size, length, length, -1)
            physical_pair = physical_h.reshape(batch_size, length, length, -1)

            phys = torch.cat([physical_pair, geom], dim=-1)
            edge_weight = torch.tanh(self.physical_head(phys).squeeze(-1)) * offdiag
            edge_weight = 0.5 * (edge_weight + edge_weight.transpose(1, 2))

            rel = coords[:, None, :, :] - coords[:, :, None, :]
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
            direction = rel / dist
            delta = (edge_weight[..., None] * direction).sum(dim=2) / max(length - 1, 1)
            delta_norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
            scale = torch.tanh(delta_norm / self.max_delta) * self.max_delta / delta_norm
            coords = coords + delta * scale
            coords = self._project_ca_bonds(coords)
            coords = coords - coords.mean(dim=1, keepdim=True)
            trajectory.append(coords)

        trajectory_tensor = torch.stack(trajectory, dim=1)
        if not return_aux:
            return trajectory_tensor

        final_geom = self._geometry_features(coords)
        final_pair = torch.cat([latent_pair, physical_pair, final_geom], dim=-1)
        distogram_logits = self.distogram_head(final_pair)
        distogram_logits = 0.5 * (distogram_logits + distogram_logits.transpose(1, 2))
        return trajectory_tensor, {"distogram_logits": distogram_logits}

    def _initial_latent_pair(self, emb: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = emb.shape
        left = emb[:, :, None, :].expand(batch_size, length, length, -1)
        right = emb[:, None, :, :].expand(batch_size, length, length, -1)
        sep = torch.arange(length, device=emb.device, dtype=emb.dtype)
        sep = (sep[None, :] - sep[:, None]).abs()
        signed_sep = torch.arange(length, device=emb.device, dtype=emb.dtype)
        signed_sep = (signed_sep[None, :] - signed_sep[:, None]) / self.max_sep
        sep = (sep / self.max_sep).clamp(max=1.0)
        pair_features = torch.cat(
            [
                left,
                right,
                sep[None, :, :, None].expand(batch_size, -1, -1, -1),
                signed_sep[None, :, :, None].expand(batch_size, -1, -1, -1),
            ],
            dim=-1,
        )
        return self.latent_init(pair_features)

    def _geometry_features(self, coords: torch.Tensor) -> torch.Tensor:
        rel = coords[:, None, :, :] - coords[:, :, None, :]
        dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
        inv_dist = 1.0 / (1.0 + dist)
        contact = torch.sigmoid((8.0 - dist) / 2.0)
        near_clash = torch.sigmoid((3.0 - dist) / 0.5)
        rbf_centers = torch.tensor([4.0, 6.0, 8.0, 12.0, 16.0], device=coords.device, dtype=coords.dtype)
        rbf = torch.exp(-((dist[..., None] - rbf_centers) ** 2) / 18.0)
        return torch.cat([dist[..., None] / 32.0, inv_dist[..., None], contact[..., None], near_clash[..., None], rbf], dim=-1)

    def _project_ca_bonds(self, coords: torch.Tensor) -> torch.Tensor:
        if self.bond_project_iters <= 0 or coords.shape[1] < 2:
            return coords

        target = coords.new_tensor(self.bond_length)
        for parity in range(self.bond_project_iters * 2):
            start = parity % 2
            if start >= coords.shape[1] - 1:
                continue
            left = coords[:, start:-1:2]
            right = coords[:, start + 1 :: 2]
            bond = right - left
            dist = torch.linalg.norm(bond, dim=-1, keepdim=True).clamp_min(1e-6)
            correction = 0.5 * (dist - target) * bond / dist
            updated = coords.clone()
            updated[:, start:-1:2] = left + correction
            updated[:, start + 1 :: 2] = right - correction
            coords = updated
        return coords


class TwoChannelTransformerFolder(nn.Module):
    """Transformer dynamics with separate latent and physical channels.

    The latent channel is free to carry bioinformatic state. The physical
    channel is the only channel decoded into coordinates, and every decoded
    state is projected back onto a coherent C-alpha chain.
    """

    def __init__(
        self,
        aa_count: int = 20,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        steps: int = 8,
        max_delta: float = 0.8,
        max_len: int = 2048,
        bond_length: float = 3.8,
        bond_project_iters: int = 2,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.max_delta = max_delta
        self.bond_length = bond_length
        self.bond_project_iters = bond_project_iters
        self.embed = nn.Embedding(aa_count, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.latent_channel = nn.Parameter(torch.randn(dim) * 0.02)
        self.physical_channel = nn.Parameter(torch.randn(dim) * 0.02)
        self.geom_embed = nn.Sequential(
            nn.Linear(9, dim),
            nn.SiLU(),
            nn.LayerNorm(dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=depth)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 3),
        )
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(dim * 4 + 2),
            nn.Linear(dim * 4 + 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 38),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def initial_backbone(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        x = torch.arange(length, device=device, dtype=torch.float32) * self.bond_length
        coords = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
        coords = coords - coords.mean(dim=0, keepdim=True)
        return coords[None, :, :].repeat(batch_size, 1, 1)

    def forward(
        self,
        seq: torch.Tensor,
        initial_coords: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, length = seq.shape
        device = seq.device
        coords = (
            self.initial_backbone(batch_size, length, device)
            if initial_coords is None
            else initial_coords - initial_coords.mean(dim=1, keepdim=True)
        )
        positions = torch.arange(length, device=device).clamp_max(self.pos_embed.num_embeddings - 1)
        base = self.embed(seq) + self.pos_embed(positions)[None]
        latent = base + self.latent_channel[None, None]
        physical = base + self.physical_channel[None, None] + self.geom_embed(self._residue_geometry(coords))
        trajectory = []
        layer_hiddens: list[torch.Tensor] = []

        for _ in range(self.steps):
            physical = physical + self.geom_embed(self._residue_geometry(coords))
            tokens = torch.cat([latent, physical], dim=1)
            mixed, step_layer_hiddens = self._mix_with_layer_hiddens(tokens)
            latent, physical = mixed[:, :length], mixed[:, length:]
            layer_hiddens = [
                torch.cat([state[:, :length], state[:, length:]], dim=-1)
                for state in step_layer_hiddens
            ]

            frames = self._ca_frames(coords)
            local_delta = torch.tanh(self.delta_head(physical)) * self.max_delta
            delta = torch.einsum("bluv,blu->blv", frames, local_delta)
            coords = coords + delta
            coords = self._project_ca_bonds(coords)
            coords = coords - coords.mean(dim=1, keepdim=True)
            trajectory.append(coords)

        trajectory_tensor = torch.stack(trajectory, dim=1)
        if not return_aux:
            return trajectory_tensor

        hidden = torch.cat([latent, physical], dim=-1)
        distogram_logits = self._distogram_logits_from_state(hidden)
        return trajectory_tensor, {
            "distogram_logits": distogram_logits,
            "hidden": hidden,
            "layer_hiddens": layer_hiddens,
        }

    def _mix_with_layer_hiddens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hiddens = []
        output = tokens
        for layer in self.mixer.layers:
            output = layer(output)
            hiddens.append(output)
        if self.mixer.norm is not None:
            output = self.mixer.norm(output)
            hiddens[-1] = output
        return output, hiddens

    def _distogram_logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = state.shape
        left = state[:, :, None, :].expand(batch_size, length, length, -1)
        right = state[:, None, :, :].expand(batch_size, length, length, -1)
        sep = torch.arange(length, device=state.device, dtype=state.dtype)
        signed_sep = (sep[None, :] - sep[:, None]) / 32.0
        abs_sep = signed_sep.abs().clamp(max=1.0)
        pair = torch.cat(
            [
                left,
                right,
                abs_sep[None, :, :, None].expand(batch_size, -1, -1, -1),
                signed_sep[None, :, :, None].expand(batch_size, -1, -1, -1),
            ],
            dim=-1,
        )
        logits = self.distogram_head(pair)
        return 0.5 * (logits + logits.transpose(1, 2))

    def _residue_geometry(self, coords: torch.Tensor) -> torch.Tensor:
        frames = self._ca_frames(coords)
        bonds_left = torch.zeros_like(coords)
        bonds_right = torch.zeros_like(coords)
        bonds_left[:, 1:] = coords[:, 1:] - coords[:, :-1]
        bonds_right[:, :-1] = coords[:, 1:] - coords[:, :-1]
        local_left = torch.einsum("blij,blj->bli", frames.transpose(-1, -2), bonds_left)
        local_right = torch.einsum("blij,blj->bli", frames.transpose(-1, -2), bonds_right)
        left_len = torch.linalg.norm(bonds_left, dim=-1, keepdim=True) / self.bond_length
        right_len = torch.linalg.norm(bonds_right, dim=-1, keepdim=True) / self.bond_length
        seq_pos = torch.linspace(-1.0, 1.0, coords.shape[1], device=coords.device, dtype=coords.dtype)
        seq_pos = seq_pos[None, :, None].expand(coords.shape[0], -1, -1)
        return torch.cat([local_left / self.bond_length, local_right / self.bond_length, left_len, right_len, seq_pos], dim=-1)

    def _ca_frames(self, coords: torch.Tensor) -> torch.Tensor:
        prev_idx = torch.arange(coords.shape[1], device=coords.device).sub(1).clamp_min(0)
        next_idx = torch.arange(coords.shape[1], device=coords.device).add(1).clamp_max(coords.shape[1] - 1)
        tangent = F.normalize(coords[:, next_idx] - coords[:, prev_idx], dim=-1, eps=1e-6)
        z_ref = torch.tensor([0.0, 0.0, 1.0], device=coords.device, dtype=coords.dtype).expand_as(tangent)
        y_ref = torch.tensor([0.0, 1.0, 0.0], device=coords.device, dtype=coords.dtype).expand_as(tangent)
        use_y = torch.abs((tangent * z_ref).sum(dim=-1, keepdim=True)) > 0.9
        ref = torch.where(use_y, y_ref, z_ref)
        normal = F.normalize(torch.cross(tangent, ref, dim=-1), dim=-1, eps=1e-6)
        binormal = F.normalize(torch.cross(tangent, normal, dim=-1), dim=-1, eps=1e-6)
        return torch.stack([tangent, normal, binormal], dim=-1)

    def _project_ca_bonds(self, coords: torch.Tensor) -> torch.Tensor:
        if self.bond_project_iters <= 0 or coords.shape[1] < 2:
            return coords

        target = coords.new_tensor(self.bond_length)
        for parity in range(self.bond_project_iters * 2):
            start = parity % 2
            if start >= coords.shape[1] - 1:
                continue
            left = coords[:, start:-1:2]
            right = coords[:, start + 1 :: 2]
            bond = right - left
            dist = torch.linalg.norm(bond, dim=-1, keepdim=True).clamp_min(1e-6)
            correction = 0.5 * (dist - target) * bond / dist
            updated = coords.clone()
            updated[:, start:-1:2] = left + correction
            updated[:, start + 1 :: 2] = right - correction
            coords = updated
        return coords


def build_model(
    architecture: str,
    steps: int,
    max_delta: float,
    pair_dim: int = 96,
    bond_project_iters: int = 2,
    transformer_dim: int = 128,
    transformer_depth: int = 4,
    transformer_heads: int = 4,
) -> nn.Module:
    if architecture == "residue":
        return RecurrentFolder(steps=steps, max_delta=max_delta)
    if architecture == "pairwise":
        return PairwiseRecurrentFolder(
            steps=steps,
            max_delta=max_delta,
            pair_dim=pair_dim,
            bond_project_iters=bond_project_iters,
        )
    if architecture == "transformer":
        return TwoChannelTransformerFolder(
            steps=steps,
            max_delta=max_delta,
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            bond_project_iters=bond_project_iters,
        )
    raise ValueError(f"Unknown architecture: {architecture}")

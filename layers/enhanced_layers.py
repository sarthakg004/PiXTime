# Audit findings:
# - In this fork, target patching is performed in models/PiXTime.py via Patch_EnEmbedding.unfold(..., step=patch_len).
# - Variable identity embeddings are represented by an nn.Embedding over variable index (iTrans_ExEmbedding.pos_embedding).
# - enc_in is inferred from dataset loaders and passed into model construction as n_vars.
# - Decoder expects a fixed patch token count (patch_num plus a global token), so adaptive outputs must be aligned.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualVariableEmbedding(nn.Module):
    """Dynamic variable embedding from identity and context.

    Purpose:
        Fuse static variable identity with a contextual representation from raw
        auxiliary series values.

    Inputs:
        var_ids: (B, M)
        aux_series: (B, M, L_in)

    Output:
        embeddings: (B, M, d_model)
    """

    def __init__(self, num_vars, d_model, context_dim=32, L_in=96):
        super().__init__()
        self.global_embed = nn.Embedding(num_vars, d_model)
        self.context_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, context_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.context_proj = nn.Linear(context_dim, d_model)
        self.L_in = L_in

    def forward(self, var_ids, aux_series):
        # var_ids: (B, M), aux_series: (B, M, L_in)
        B, M, L = aux_series.shape

        global_emb = self.global_embed(var_ids)  # (B, M, d_model)

        if M == 0:
            return global_emb

        x = aux_series.reshape(B * M, 1, L)      # (B*M, 1, L)
        ctx = self.context_encoder(x).squeeze(-1)  # (B*M, context_dim)
        ctx = self.context_proj(ctx)               # (B*M, d_model)
        ctx = ctx.reshape(B, M, -1)                # (B, M, d_model)

        return global_emb + ctx


class VariableRelationLayer(nn.Module):
    """Self-attention over variable tokens.

    Input:
        var_tokens: (B, M, d_model)

    Output:
        var_tokens: (B, M, d_model)
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, var_tokens):
        attn_out, _ = self.attn(var_tokens, var_tokens, var_tokens)
        return self.norm(var_tokens + self.dropout(attn_out))


class AdaptivePatchEmbedding(nn.Module):
    """Adaptive change-point patch embedding with fixed token count alignment.

    Input:
        x: (B, L_in)
        target_n_patches: int

    Output:
        embeddings: (B, target_n_patches, d_model)
    """

    def __init__(
        self,
        patch_len,
        d_model,
        dropout=0.1,
        threshold_factor=1.5,
        min_patch=4,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.min_patch = min_patch
        self.threshold_factor = threshold_factor
        self.proj = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _get_boundaries(self, x_single):
        """Return list of (start, end) segment boundaries for one sample."""
        L = len(x_single)
        if L <= 1:
            return [(0, max(1, L))]

        diff = torch.abs(x_single[1:] - x_single[:-1])
        threshold = diff.mean() * self.threshold_factor

        change_pts = (diff > threshold).nonzero(as_tuple=True)[0] + 1
        change_pts = change_pts.tolist()

        boundaries = [0]
        for cp in change_pts:
            if cp - boundaries[-1] >= self.min_patch:
                boundaries.append(cp)
        boundaries.append(L)

        segments = [
            (boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
            if boundaries[i + 1] > boundaries[i]
        ]

        if not segments:
            segments = [(0, L)]
        return segments

    def forward(self, x, target_n_patches):
        B, _ = x.shape
        all_embeddings = []

        for b in range(B):
            segs = self._get_boundaries(x[b])
            patch_embs = []

            for s, e in segs:
                seg = x[b, s:e]
                seg = seg.unsqueeze(0).unsqueeze(0)  # (1, 1, seg_len)
                seg = F.interpolate(
                    seg,
                    size=self.patch_len,
                    mode='linear',
                    align_corners=False,
                ).reshape(self.patch_len)
                patch_embs.append(self.proj(seg))

            patch_embs = torch.stack(patch_embs, dim=0)
            n_tokens = patch_embs.shape[0]

            if n_tokens < target_n_patches:
                pad = torch.zeros(
                    target_n_patches - n_tokens,
                    patch_embs.shape[-1],
                    device=x.device,
                    dtype=patch_embs.dtype,
                )
                patch_embs = torch.cat([patch_embs, pad], dim=0)
            elif n_tokens > target_n_patches:
                patch_embs = patch_embs[:target_n_patches]

            all_embeddings.append(patch_embs)

        out = torch.stack(all_embeddings, dim=0)
        return self.norm(self.dropout(out))


class MultiScalePatchEmbedding(nn.Module):
    """Multi-scale patch encoder with token-length alignment and fusion.

    Input:
        x: (B, L_in)
        target_n_patches: optional int

    Output:
        fused: (B, N_p, d_model)
    """

    def __init__(self, d_model, patch_sizes=(8, 16, 32), dropout=0.1):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.projs = nn.ModuleList([nn.Linear(p, d_model) for p in patch_sizes])
        self.fusion = nn.Linear(d_model * len(patch_sizes), d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _patchify(self, x, patch_size):
        B, L = x.shape
        n = max(1, L // patch_size)
        x = x[:, :n * patch_size]
        return x.reshape(B, n, patch_size)

    def _align_tokens(self, emb, token_count):
        if emb.shape[1] == token_count:
            return emb
        emb = emb.permute(0, 2, 1)
        emb = F.interpolate(emb, size=token_count, mode='linear', align_corners=False)
        return emb.permute(0, 2, 1)

    def forward(self, x, target_n_patches=None):
        B, L = x.shape
        valid = []
        for p, proj in zip(self.patch_sizes, self.projs):
            if L < p:
                continue
            patches = self._patchify(x, p)
            valid.append(proj(patches))

        if not valid:
            # Graceful fallback when sequence length is smaller than all configured patch sizes.
            p = self.patch_sizes[0]
            resized = F.interpolate(
                x.unsqueeze(1),
                size=p,
                mode='linear',
                align_corners=False,
            ).squeeze(1).unsqueeze(1)
            valid = [self.projs[0](resized)]

        ref_n = valid[0].shape[1]
        aligned = [self._align_tokens(v, ref_n) for v in valid]

        # If fewer scales are active, pad feature dimension to keep fusion input fixed.
        if len(aligned) < len(self.patch_sizes):
            missing = len(self.patch_sizes) - len(aligned)
            zeros = torch.zeros(B, ref_n, self.fusion.in_features // len(self.patch_sizes), device=x.device)
            aligned.extend([zeros for _ in range(missing)])

        fused = torch.cat(aligned, dim=-1)
        fused = self.fusion(fused)

        if target_n_patches is not None:
            fused = self._align_tokens(fused, target_n_patches)

        return self.norm(self.dropout(fused))

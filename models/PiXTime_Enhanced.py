# Audit findings:
# - Baseline target patch path is models/PiXTime.py: Patch_EnEmbedding.forward using unfold over time.
# - Baseline variable identity embedding is iTrans_ExEmbedding.pos_embedding (equivalent to a VE table).
# - enc_in is inferred in dataset loaders and passed to model construction as n_vars.
# - Decoder consumes fixed-length patch tokens with a final global token, so enhanced patch paths must keep this shape.

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import layers.Transformer_EncDec as Transformer_EncDec

from layers.enhanced_layers import (
    ContextualVariableEmbedding,
    VariableRelationLayer,
    AdaptivePatchEmbedding,
    MultiScalePatchEmbedding,
)


class Patch_EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class iTrans_ExEmbedding(nn.Module):
    def __init__(
        self,
        seq_len=96,
        n_vars=7,
        d_model=128,
        dropout=0.1,
        factor=3,
        n_heads=8,
        d_ff=128,
        e_layers=1,
        activation='gelu',
        use_contextual_var_emb=False,
        contextual_var_emb=None,
        use_var_relation=False,
        var_relation=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.enc_embedding = DataEmbedding_inverted(c_in=seq_len, d_model=d_model, dropout=dropout)

        # Baseline-compatible variable identity embedding (VE table equivalent).
        self.VE_table = nn.Embedding(n_vars, d_model)

        self.use_contextual_var_emb = use_contextual_var_emb
        self.contextual_var_emb = contextual_var_emb

        self.use_var_relation = use_var_relation
        self.var_relation = var_relation

        self.encoder = Transformer_EncDec.Encoder(
            [
                Transformer_EncDec.EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

    def forward(self, x_enc):
        # x_enc: (B, M, L_in)
        enc_out = self.enc_embedding(x_enc, None)  # (B, M, d_model)
        B, M, _ = enc_out.shape
        var_ids = torch.arange(M, device=x_enc.device).unsqueeze(0).expand(B, M)
        aux_series = x_enc.permute(0, 2, 1) if x_enc.shape[1] == self.seq_len else x_enc

        # [ENHANCED: Shashank Imp. 1]
        if self.use_contextual_var_emb and self.contextual_var_emb is not None:
            var_emb = self.contextual_var_emb(var_ids, aux_series)
        else:
            var_emb = self.VE_table(var_ids)

        aux_tokens = enc_out + var_emb

        # [ENHANCED: Shashank Imp. 2]
        if self.use_var_relation and self.var_relation is not None:
            aux_tokens = self.var_relation(aux_tokens)

        aux_tokens, _ = self.encoder(aux_tokens)
        return aux_tokens


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x[:, :, :, :-1])
        x = self.linear(x)
        x = self.dropout(x)
        return x


class XDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class XDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, _, D = cross.shape
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(
            self.cross_attention(x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0]
        )
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])
        ).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):
    def __init__(
        self,
        configs,
        use_contextual_var_emb=False,
        use_var_relation=False,
        use_adaptive_patch=False,
        use_multiscale_patch=False,
    ):
        super().__init__()

        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.features = getattr(configs, 'features', 'M')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = 1
        self.patch_len = configs.patch_len

        if self.seq_len % self.patch_len != 0:
            raise ValueError(f"seq_len ({self.seq_len}) must be divisible by patch_len ({self.patch_len})")

        self.patch_num = self.seq_len // self.patch_len
        self.n_vars = 1 if self.features == 'MS' else configs.enc_in

        d_model = configs.d_model
        dropout = configs.dropout

        self.patch_embedding = Patch_EnEmbedding(self.n_vars, d_model, self.patch_len, dropout)

        # --- Variable embedding enhancement (Shashank Imp. 1) ---
        self.use_contextual_var_emb = use_contextual_var_emb
        if use_contextual_var_emb:
            self.contextual_var_emb = ContextualVariableEmbedding(
                num_vars=configs.enc_in,
                d_model=configs.d_model,
                L_in=configs.seq_len,
            )
        else:
            self.contextual_var_emb = None

        # --- Variable relation layer (Shashank Imp. 2) ---
        self.use_var_relation = use_var_relation
        if use_var_relation:
            self.var_relation = VariableRelationLayer(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                dropout=configs.dropout,
            )
        else:
            self.var_relation = None

        self.ex_embedding = iTrans_ExEmbedding(
            seq_len=configs.seq_len,
            n_vars=configs.enc_in,
            d_model=configs.d_model,
            dropout=configs.dropout,
            factor=configs.factor,
            n_heads=configs.n_heads,
            d_ff=configs.en_d_ff,
            e_layers=configs.en_layers,
            use_contextual_var_emb=self.use_contextual_var_emb,
            contextual_var_emb=self.contextual_var_emb,
            use_var_relation=self.use_var_relation,
            var_relation=self.var_relation,
        )

        # --- Adaptive patch embedding (Tilak Imp. 1) ---
        self.use_adaptive_patch = use_adaptive_patch
        if use_adaptive_patch:
            self.adaptive_patch_emb = AdaptivePatchEmbedding(
                patch_len=configs.patch_len,
                d_model=configs.d_model,
                dropout=configs.dropout,
            )

        # --- Multi-scale patch embedding (Tilak Imp. 2) ---
        self.use_multiscale_patch = use_multiscale_patch
        if use_multiscale_patch:
            self.multiscale_patch_emb = MultiScalePatchEmbedding(
                d_model=configs.d_model,
                patch_sizes=(8, 16, 32),
                dropout=configs.dropout,
            )

        self.cross_decoder = XDecoder(
            [
                XDecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.de_d_ff,
                    dropout=configs.dropout,
                    activation='gelu',
                )
                for _ in range(configs.de_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(self.head_nf, self.pred_len, head_dropout=configs.dropout)

    def _enhanced_patch_tokens(self, x_vars):
        """Build enhanced patch tokens and append the baseline global token.

        Args:
            x_vars: (B, n_vars, L)

        Returns:
            patch_embed: (B*n_vars, patch_num+1, d_model)
            n_vars: int
        """
        B, n_vars, _ = x_vars.shape
        target_n_patches = self.patch_num
        var_tokens = []

        for idx in range(n_vars):
            series = x_vars[:, idx, :]  # (B, L)

            # [ENHANCED: Tilak Imp. 1]
            if self.use_adaptive_patch:
                emb = self.adaptive_patch_emb(series, target_n_patches)
            # [ENHANCED: Tilak Imp. 2]
            elif self.use_multiscale_patch:
                emb = self.multiscale_patch_emb(series, target_n_patches=target_n_patches)
            else:
                raise RuntimeError('Enhanced patch path called without enhancement flags enabled.')

            emb = emb + self.patch_embedding.position_embedding(emb)
            var_tokens.append(emb)

        patch_tokens = torch.stack(var_tokens, dim=1)  # (B, n_vars, patch_num, d_model)
        glb = self.patch_embedding.glb_token[:, :n_vars, :, :].repeat(B, 1, 1, 1)
        patch_tokens = torch.cat([patch_tokens, glb], dim=2)  # (B, n_vars, patch_num+1, d_model)
        patch_tokens = patch_tokens.reshape(B * n_vars, patch_tokens.shape[2], patch_tokens.shape[3])
        patch_tokens = self.patch_embedding.dropout(patch_tokens)
        return patch_tokens, n_vars

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if self.use_adaptive_patch or self.use_multiscale_patch:
            # [ENHANCED: Tilak improvements]
            patch_embed, n_vars = self._enhanced_patch_tokens(x_enc[:, :, -1].unsqueeze(1))
        else:
            patch_embed, n_vars = self.patch_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))

        ex_embed = self.ex_embedding(x_enc[:, :, :-1])

        out = self.cross_decoder(patch_embed, ex_embed)
        out = torch.reshape(out, (-1, n_vars, out.shape[-2], out.shape[-1]))
        out = out.permute(0, 1, 3, 2)

        out = self.head(out)
        out = out.permute(0, 2, 1)

        if self.use_norm:
            out = out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            out = out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return out

    def forecast_multi(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if self.use_adaptive_patch or self.use_multiscale_patch:
            # [ENHANCED: Tilak improvements]
            patch_embed, n_vars = self._enhanced_patch_tokens(x_enc.permute(0, 2, 1))
        else:
            patch_embed, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1))

        ex_embed = self.ex_embedding(x_enc)

        out = self.cross_decoder(patch_embed, ex_embed)
        out = torch.reshape(out, (-1, n_vars, out.shape[-2], out.shape[-1]))
        out = out.permute(0, 1, 3, 2)

        out = self.head(out)
        out = out.permute(0, 2, 1)

        if self.use_norm:
            out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.features == 'M':
            dec_out = self.forecast_multi(x_enc)
            return dec_out[:, -self.pred_len:, :]

        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]

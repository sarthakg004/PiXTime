<!--
Audit findings:
- Baseline PiXTime uses fixed non-overlapping temporal patches and variable-index embeddings.
- exogenous variables are encoded through DataEmbedding_inverted + variable identity embedding.
- Decoder requires a fixed patch token count and a final global token.
-->

# PiXTime Architectural Improvements

This document summarizes four additive, flag-controlled enhancements I implemented on top of the baseline PiXTime model. All flags default to disabled, so baseline behavior remains available.

## 1) Contextual Variable Embedding (Shashank Imp. 1)
I introduced a dynamic variable representation that augments each variable's identity embedding with context extracted directly from that variable's recent time series. Concretely, I pass each auxiliary variable sequence through a lightweight 1D CNN encoder, project the context to model dimension, and sum it with the global identity embedding. This gives the model a variable representation that can adapt by sample rather than remaining purely static.

Flag: --use_contextual_var_emb

## 2) Variable Relation Layer (Shashank Imp. 2)
I added a dedicated variable-interaction module based on multi-head self-attention across the variable axis. After initial auxiliary-variable token construction, this layer lets each variable token attend to the others before entering the encoder stack. The design is residual plus LayerNorm, so it is shape-preserving and easy to compose with the baseline pipeline.

Flag: --use_var_relation

## 3) Adaptive Patch Embedding (Tilak Imp. 1)
I implemented change-point-aware temporal segmentation for the target series. Instead of fixed windows only, this module first detects abrupt local differences, forms variable-length segments, and then resizes each segment to patch length before projection. To preserve decoder compatibility, I always align adaptive tokens to the baseline patch count and append the same global token structure expected by the existing decoder.

Flag: --use_adaptive_patch

Hyperparameters:
- threshold_factor: 1.5
- min_patch: 4
- patch_len: inherited from run args (default 16 in this repo)

## 4) Multi-Scale Patch Embedding (Tilak Imp. 2)
I added a multi-resolution temporal encoder that patchifies each target series at patch sizes (8, 16, 32), projects each scale independently, aligns token counts by interpolation, concatenates scale features, and fuses to d_model. This enables the model to mix local and coarser temporal patterns while still outputting the fixed token shape expected downstream.

Flag: --use_multiscale_patch

Hyperparameters:
- patch_sizes: (8, 16, 32)

## CLI Usage

Baseline:
python run.py --model PiXTime ...

Enhanced model (toggle any subset):
python run.py --model PiXTime_Enhanced ... --use_contextual_var_emb --use_var_relation --use_adaptive_patch --use_multiscale_patch

## Current Key Results

Source: results/summary.txt

| Dataset | Horizon | baseline MSE | contextual_ve | var_relation | adaptive_patch | multiscale_patch | all_improvements |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.3994 | 0.3942 | 0.3921 | 0.5906 | 0.3972 | 0.5890 |

Observed delta vs baseline (MSE):
- contextual_ve: +1.31% improvement
- var_relation: +1.83% improvement
- adaptive_patch: -47.86% (worse)
- multiscale_patch: +0.56% improvement
- all_improvements: -47.45% (worse)

## Notes

- All improvements are additive and independently switchable.
- Baseline path remains available with default flags disabled.
- Enhanced comparison script: python evaluate_improvements.py
- Grid launcher script: ./pixtime_enhanced_run.sh

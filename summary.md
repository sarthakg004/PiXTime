# PiXTime: Project Summary

## Overview

**PiXTime** is a deep learning model for **long-term time series forecasting** designed specifically for **federated learning scenarios with heterogeneous data structures across nodes**. The project implements PiXTime along with several state-of-the-art baseline models for comprehensive comparison.

---

## Project Structure

```
PiXTime/
├── models/                    # Model implementations
│   ├── PiXTime.py           # Main proposed model
│   ├── iTransformer.py      # Inverted Transformer baseline
│   ├── PatchTST.py          # PatchTST baseline
│   ├── TimeXer.py           # TimeXer baseline
│   ├── DLinear.py           # DLinear baseline
│   └── Autoformer.py        # Autoformer baseline
├── layers/                    # Shared neural network layers
│   ├── Transformer_EncDec.py    # Encoder/Decoder architectures
│   ├── SelfAttention_Family.py  # Attention mechanisms
│   ├── Embed.py                 # Embedding layers
│   └── ...
├── dataset/                   # Data loading components
│   ├── data_factory.py        # Data provider factory
│   └── data_loader.py         # Dataset implementations
├── utils/                     # Utility functions
│   ├── metrics.py             # Evaluation metrics (MSE, MAE, etc.)
│   ├── tools.py               # Training utilities
│   └── timefeatures.py        # Time feature encoding
├── evaluation/                # Experimental results
│   ├── PiXTime/               # PiXTime results
│   ├── iTransformer/          # iTransformer results
│   ├── PatchTST/              # PatchTST results
│   └── DLinear/               # DLinear results
├── run.py                     # Main training script
├── pixtime_run.sh             # PiXTime experiment configs
├── patchtst_run.sh            # PatchTST experiment configs
├── itrans_run.sh              # iTransformer experiment configs
├── dlinear_run.sh             # DLinear experiment configs
└── timexer_run.sh             # TimeXer experiment configs
```

---

## Core Method: PiXTime Architecture

### Key Innovation
PiXTime addresses the challenge of **federated time series forecasting with heterogeneous data structures** across different nodes. The architecture combines:

1. **Patch-based Encoding**: Divides time series into patches for local pattern extraction
2. **Inverted Transformer Encoding**: Uses iTransformer-style encoding for cross-variable attention
3. **Cross-Attention Decoder**: Enables information fusion between patch embeddings and variable embeddings

### Architecture Components

#### 1. Patch_EnEmbedding (`Patch_EnEmbedding`)
- **Purpose**: Embeds the target variable (or all variables in multi-variate mode) using patch-based tokenization
- **Process**:
  - Splits input sequence into non-overlapping patches of size `patch_len`
  - Applies linear projection to map patches to `d_model` dimensions
  - Adds positional embeddings
  - Appends a learnable global token (`glb_token`) for global information aggregation
- **Output**: Patch embeddings of shape `[batch_size * n_vars, patch_num + 1, d_model]`

#### 2. iTrans_ExEmbedding (`iTrans_ExEmbedding`)
- **Purpose**: Encodes external variables (all variables except the target) using inverted transformer architecture
- **Process**:
  - Transposes input to treat variables as tokens (iTransformer approach)
  - Applies inverted data embedding
  - Adds variable-specific positional embeddings
  - Passes through transformer encoder layers with multi-head attention
- **Output**: Variable embeddings of shape `[batch_size, n_vars, d_model]`

#### 3. XDecoder (`XDecoder`)
- **Purpose**: Cross-attention decoder that fuses patch and variable information
- **XDecoderLayer Components**:
  - **Self-Attention**: Captures relationships among patches
  - **Cross-Attention**: Global token attends to external variable embeddings
  - **Feed-Forward Network**: 1D convolutions for local feature refinement
- **Key Design**: Only the global token participates in cross-attention with external variables, reducing computational complexity

#### 4. FlattenHead
- **Purpose**: Projects decoder output to prediction horizon
- **Process**: Flattens patch dimensions and applies linear projection to `pred_len`

### Forward Flow

```
Input: x_enc [B, seq_len, n_vars]

1. Normalization (Non-stationary Transformer style):
   - Compute mean and std across time dimension
   - Normalize input

2. Patch Embedding (for target variable):
   - Extract target: x_enc[:, :, -1]
   - Patch and embed → [B * n_vars, patch_num + 1, d_model]

3. External Variable Embedding:
   - Process all variables → [B, n_vars, d_model]

4. Cross-Attention Decoder:
   - Self-attention among patches
   - Cross-attention: global token ↔ external variables
   - Feed-forward processing

5. Output Projection:
   - Reshape and flatten
   - Linear projection to pred_len

6. De-normalization:
   - Restore original scale

Output: [B, pred_len, n_vars]
```

### Feature Modes
- **M (Multivariate)**: Predicts all variables simultaneously using `forecast_multi()`
- **MS (Multivariate-Single)**: Uses all variables to predict only the target variable
- **S (Single)**: Univariate forecasting (not fully supported in PiXTime)

---

## Baseline Models

### 1. iTransformer
- **Paper**: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (2023)
- **Key Idea**: Inverts the dimensions to apply attention across variables instead of time steps
- **Architecture**: Encoder-only with inverted embedding + variable-wise attention

### 2. PatchTST
- **Paper**: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (2022)
- **Key Idea**: Patches time series into subseries-level patches, applies channel-independent Transformer
- **Architecture**: Patch embedding + Transformer encoder + Flatten head

### 3. TimeXer
- **Key Idea**: Combines patch-based encoding with cross-attention for external variables
- **Architecture**: Similar to PiXTime but with encoder-only design

### 4. DLinear
- **Paper**: "Are Transformers Effective for Time Series Forecasting?" (2022)
- **Key Idea**: Simple linear model with decomposition (seasonal + trend)
- **Architecture**: Series decomposition + two linear layers

### 5. Autoformer
- **Paper**: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (2021)
- **Key Idea**: Uses auto-correlation mechanism and series decomposition

---

## Datasets

The project supports 8 benchmark datasets for time series forecasting:

| Dataset | Type | Variables | Description |
|---------|------|-----------|-------------|
| **ETTh1** | Hourly | 7 | Electricity Transformer Temperature (hourly) |
| **ETTh2** | Hourly | 7 | Electricity Transformer Temperature (hourly) |
| **ETTm1** | 15-min | 7 | Electricity Transformer Temperature (15-minute) |
| **ETTm2** | 15-min | 7 | Electricity Transformer Temperature (15-minute) |
| **electricity** | Hourly | 321 | UCI Electricity Consumption |
| **exchange_rate** | Daily | 8 | Currency exchange rates |
| **traffic** | Hourly | 862 | PeMS Traffic flow data |
| **weather** | 10-min | 21 | Weather station measurements |

### Data Splits
- **ETT datasets**: 12 months train / 4 months validation / 4 months test
- **Custom datasets**: 70% train / 10% validation / 20% test

### Preprocessing
- StandardScaler normalization (fit on train, apply to all)
- Time feature encoding (month, day, weekday, hour, minute)

---

## Experimental Setup

### Prediction Settings
- **Input Length (seq_len)**: 96
- **Prediction Lengths (pred_len)**: {96, 192, 336, 720}
- **Patch Length (patch_len)**: 16

### Hyperparameters

#### PiXTime Configuration
| Dataset | d_model | en_d_ff | de_d_ff | en_layers | de_layers |
|---------|---------|---------|---------|-----------|-----------|
| ETTh1 | 512 | 2048 | 2048 | 1 | 1 |
| ETTh2 | 128-512 | 512-2048 | 512-2048 | 1 | 1 |
| ETTm1/m2 | 256-512 | 1024-2048 | 1024-2048 | 1 | 1 |
| electricity | 512 | 2048 | 2048 | 2 | 2 |
| exchange_rate | 64-128 | 256-512 | 256-512 | 1 | 1 |
| traffic | 512 | 2048 | 2048 | 1 | 1 |
| weather | 512 | 2048 | 2048 | 1 | 1 |

#### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Batch Size**: 32
- **Epochs**: 10
- **Loss Function**: MSE (Mean Squared Error)
- **Learning Rate Scheduler**: type1 (halves every epoch)

### Evaluation Metrics
- **MSE** (Mean Squared Error): Primary metric
- **MAE** (Mean Absolute Error): Secondary metric
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **MSPE** (Mean Squared Percentage Error)

---

## Results Summary

### Sample Results (ETTh1, pred_len=96)

| Model | MSE | MAE |
|-------|-----|-----|
| **PiXTime** | **0.3737** | **0.3937** |
| iTransformer | 0.3932 | 0.4086 |
| DLinear | 0.3959 | 0.4104 |

### Key Findings
1. **PiXTime achieves competitive performance** across multiple datasets and prediction horizons
2. **Cross-attention mechanism** effectively leverages external variable information
3. **Patch-based encoding** captures local temporal patterns efficiently
4. **Lightweight decoder design** (1-2 layers) is sufficient for forecasting tasks

### Result Files
Results are stored in `evaluation/{model_name}/` with naming convention:
```
{model}{data}PatL{patch_len}PreL{pred_len}Fea{features}DM{d_model}ED{en_d_ff}DD{de_d_ff}EL{en_layers}DL{de_layers}.txt
```

---

## Code Flow

### Training Pipeline (`run.py`)

```python
1. Parse arguments and configure hyperparameters
2. Load data via data_provider():
   - Returns DataLoader and number of variables (enc_in)
3. Initialize model based on args.model:
   - PiXTime, iTransformer, PatchTST, TimeXer, DLinear, Autoformer
4. Training loop:
   - Forward pass: model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
   - Compute MSE loss
   - Backpropagation and optimization
   - Learning rate adjustment
5. Validation loop (optional, for monitoring)
6. Testing loop:
   - Generate predictions
   - Compute metrics (MSE, MAE, RMSE, MAPE, MSPE)
   - Save results to evaluation file
```

### Data Loading Pipeline (`dataset/`)

```python
1. data_factory.data_provider():
   - Selects appropriate Dataset class based on dataset name
   - Configures DataLoader with batch_size and shuffle

2. Dataset classes (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom):
   - __read_data__(): Load CSV, apply StandardScaler, extract time features
   - __getitem__(): Return (seq_x, seq_y, seq_x_mark, seq_y_mark)
   - Sliding window approach for sampling
```

### Model Forward Pass (PiXTime)

```python
1. Input normalization (mean, std)
2. Patch embedding of target variable
3. iTransformer encoding of all variables
4. Cross-attention decoder:
   - Self-attention on patches
   - Cross-attention between global token and variable embeddings
   - Feed-forward processing
5. Flatten and project to prediction length
6. De-normalization
```

---

## Key Implementation Details

### 1. Non-Stationary Transformer Normalization
All models use the normalization technique from Non-stationary Transformer:
```python
means = x_enc.mean(1, keepdim=True).detach()
x_enc = x_enc - means
stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
x_enc /= stdev
# ... model forward ...
# De-normalization
out = out * stdev + means
```

### 2. Patch Embedding
- Uses `unfold` operation for non-overlapping patches
- Linear projection + positional encoding
- Global token appended for cross-attention

### 3. Inverted Embedding (iTransformer style)
- Transposes input from `[B, T, N]` to `[B, N, T]`
- Treats variables as tokens instead of time steps
- Enables efficient cross-variable attention

### 4. Cross-Attention Design
- Only the global token (last position) attends to external variables
- Reduces complexity from O(N²) to O(N)
- External variables serve as keys and values

---

## Future Scope

### Potential Improvements

1. **Federated Learning Implementation**
   - Current implementation is centralized
   - Future work: Add federated averaging (FedAvg) or personalized federated learning
   - Handle heterogeneous data structures across clients

2. **Heterogeneity Handling**
   - Different clients may have different variable sets
   - Implement variable alignment or partial participation mechanisms
   - Explore domain adaptation techniques

3. **Model Compression**
   - Knowledge distillation for edge deployment
   - Quantization and pruning for efficiency
   - Lightweight decoder variants

4. **Extended Tasks**
   - Imputation: Fill missing values in time series
   - Anomaly Detection: Identify unusual patterns
   - Classification: Time series classification
   - Probabilistic Forecasting: Uncertainty quantification

5. **Advanced Architectures**
   - Mixture of Experts (MoE) for heterogeneous clients
   - Graph Neural Networks for variable relationships
   - Hierarchical attention mechanisms

6. **Real-World Deployment**
   - Online learning capabilities
   - Concept drift adaptation
   - Streaming data support

### Research Directions

1. **Personalized Federated Learning**: Adapt model to each client's local patterns while maintaining global knowledge
2. **Communication Efficiency**: Reduce communication overhead in federated settings
3. **Privacy Preservation**: Differential privacy and secure aggregation
4. **Cross-Domain Transfer**: Transfer knowledge across different application domains

---

## Usage Instructions

### Setup
```bash
pip install torch numpy pandas scikit-learn
```

### Prepare Datasets
Place datasets in `dataset/` folder:
- ETT-small/ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
- electricity/electricity.csv
- exchange_rate/exchange_rate.csv
- traffic/traffic.csv
- weather/weather.csv

### Run Experiments

**Run PiXTime:**
```bash
./pixtime_run.sh
```

**Run specific configuration:**
```bash
python run.py --model PiXTime --data ETTh1 --features M --seq_len 96 --pred_len 96 \
              --d_model 512 --en_d_ff 2048 --de_d_ff 2048 --en_layers 1 --de_layers 1
```

**Available models:**
- `PiXTime`: Proposed model
- `iTransformer`: Inverted Transformer baseline
- `PatchTST`: Patch-based Transformer baseline
- `TimeXer`: Time series transformer with cross-attention
- `DLinear`: Decomposition Linear baseline
- `Autoformer`: Auto-correlation Transformer

### View Results
Results are saved in `evaluation/{model_name}/` as text files containing MSE and MAE values.

---

## Dependencies

- Python 3.7+
- PyTorch 1.12+
- NumPy
- Pandas
- Scikit-learn
- einops (for attention operations)
- reformer_pytorch (optional, for Reformer attention)

---

## Citation

If you use this code or refer to PiXTime, please cite:

```bibtex
@article{pixtime2024,
  title={PiXTime: A Model for Federated Time Series Forecasting with Heterogeneous Data Structures Across Nodes},
  year={2024}
}
```

---

## License

This project is licensed under the terms specified in the LICENSE file.

---

## Acknowledgments

This implementation builds upon several open-source time series forecasting frameworks and baseline models:
- iTransformer: https://arxiv.org/abs/2310.06625
- PatchTST: https://arxiv.org/abs/2211.14730
- DLinear: https://arxiv.org/abs/2205.13504
- Autoformer: https://arxiv.org/abs/2106.13008

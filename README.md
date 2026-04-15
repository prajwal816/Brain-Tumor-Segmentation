# 🧠 Brain Tumor Classification using Differential Equation Modeling

## Binary Classification: Benign (LGG) vs Malignant (HGG) Brain Tumors

**Dataset:** [BraTS 2020 Training Data (Kaggle)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

---

## Pipeline Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  MRI Brain   │──▶│  Grayscale   │──▶│ Resize 10×10 │──▶│  Row-wise    │
│  Image (4ch) │   │  Conversion  │   │              │   │  Mean → 1D   │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                │
                                                                ▼
┌──────────────┐   ┌──────────────┐   ┌─────────────────────────────────┐
│   Output:    │   │  CNN         │   │  Differential Modeling          │
│   Benign /   │◀──│  Classifier  │◀──│  (DE/IDE with Euler & RK4)     │
│   Malignant  │   │              │   │  90 features extracted          │
└──────────────┘   └──────────────┘   └─────────────────────────────────┘
```

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Data Loading & Exploration](01_Data_Loading_and_Exploration.ipynb) | Load BraTS 2020 dataset, visualize MRI modalities, analyze tumor grades |
| 02 | [Preprocessing Pipeline](02_Preprocessing_Pipeline.ipynb) | Grayscale conversion → Resize to 10×10 → Row-wise mean → 1D signal |
| 03 | [Differential Equation Modeling](03_Differential_Equation_Modeling.ipynb) | ODE/IDE model fitting, Euler & RK4 numerical solving, feature extraction |
| 04 | [CNN Classification Model](04_CNN_Classification_Model.ipynb) | 1D CNN architecture, training with callbacks, quick evaluation |
| 05 | [Evaluation & Results](05_Evaluation_and_Results.ipynb) | ROC/PR curves, confusion matrix, Euler vs RK4 method comparison |
| 06 | [End-to-End Pipeline](06_End_to_End_Pipeline.ipynb) | Self-contained demo: complete pipeline in a single notebook |

## Mathematical Background

### Ordinary Differential Equation (ODE)
```
dy/dt = α·y + β·t + γ
```
Parameters (α, β, γ) fitted via least squares regression.

### Integro-Differential Equation (IDE)
```
dy/dt = f(y, t) + λ ∫₀ᵗ K(t,s)·y(s) ds
```
Where K(t,s) = e^(-μ(t-s)) is an exponential decay kernel.

### Numerical Solvers
- **Euler's Method** (1st order): y_{n+1} = y_n + h·f(y_n, t_n)
- **RK4 Method** (4th order): Uses weighted average of 4 slope evaluations per step

### Feature Vector (90 features per sample)
| Group | Count | Description |
|-------|-------|-------------|
| ODE Coefficients | 3 | α, β, γ from fitted model |
| Euler ODE Trajectory | 10 | Predicted signal |
| RK4 ODE Trajectory | 10 | Predicted signal |
| Euler ODE Residuals | 10 | Original - Euler |
| RK4 ODE Residuals | 10 | Original - RK4 |
| Euler IDE Trajectory | 10 | IDE solution (Euler) |
| RK4 IDE Trajectory | 10 | IDE solution (RK4) |
| Euler IDE Residuals | 10 | Original - Euler IDE |
| RK4 IDE Residuals | 10 | Original - RK4 IDE |
| Statistical Features | 7 | RMSE, energy, slope, curvature |

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) and extract to:
```
./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
```

Or set the Kaggle path in the notebooks:
```python
DATA_DIR = '/kaggle/input/brats2020-training-data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
```

### 3. Run Notebooks
Execute notebooks in order (01 → 05), or run Notebook 06 for a self-contained demo.

> **Note:** All notebooks include a **synthetic data fallback** that generates BraTS-like data when the real dataset is not available. This allows running the complete pipeline without downloading the dataset.

## CNN Architecture

```
Input (10, 9) - 10 timesteps, 9 feature channels
    │
    ├── Conv1D(64, k=3) + BatchNorm + ReLU
    ├── Conv1D(128, k=3) + BatchNorm + ReLU + Dropout(0.3)
    ├── Conv1D(256, k=3) + BatchNorm + ReLU + Dropout(0.3)
    ├── GlobalAveragePooling1D
    ├── Dense(128) + ReLU + Dropout(0.5)
    ├── Dense(64) + ReLU + Dropout(0.3)
    └── Dense(1) + Sigmoid → Benign/Malignant
```

## Project Structure
```
Brain-Tumor-Segmentation/
├── 01_Data_Loading_and_Exploration.ipynb
├── 02_Preprocessing_Pipeline.ipynb
├── 03_Differential_Equation_Modeling.ipynb
├── 04_CNN_Classification_Model.ipynb
├── 05_Evaluation_and_Results.ipynb
├── 06_End_to_End_Pipeline.ipynb
├── README.md
├── requirements.txt
├── processed_data/          # Generated intermediate files
├── models/                  # Saved models
└── results/                 # Evaluation outputs
```

## References

1. B. H. Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE TMI, 2015
2. S. Bakas et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 2017
3. S. Bakas et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv:1811.02629, 2018

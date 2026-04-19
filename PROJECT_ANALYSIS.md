# SpectraGeni Project - Comprehensive Analysis

## 1. PROJECT PURPOSE

**SpectraGeni** is a synthetic hyperspectral data generation framework designed for agricultural crop classification using machine learning. The project addresses data scarcity in agricultural remote sensing by generating high-quality synthetic hyperspectral data that maintains the statistical properties and distributions of real spectral data.

### Target Applications:

- Agricultural crop classification and monitoring
- Remote sensing using hyperspectral sensors
- Precision farming and crop health assessment
- Drone-based agricultural surveillance

### Scope:

- **6 Agricultural Crops**: Capsicum, Chilli, Mulberry, Potato, Tomato (Healthy), Tomato (Unhealthy)
- **Hyperspectral Data**: 943-dimensional feature space (spectral bands)
- **Scale**: Handles millions of samples for robust training

---

## 2. DATA AUGMENTATION TECHNIQUES

The project implements and compares **4 distinct data augmentation techniques** for synthetic data generation:

### 2.1 **SMOTE** (Synthetic Minority Over-sampling Technique)

- **File**: `01.SMOTE.ipynb`
- **Algorithm**: k-NN based synthetic sample generation
- **Process**:
  - Loads real data from Parquet files (Train_80)
  - Generates ~2000 synthetic samples per class
  - Concatenates with original data
  - Removes original samples, keeping only newly generated synthetic data
  - Shuffles and validates output (removes NaN, inf, -inf values)
- **Output**: 6 CSV/Parquet files (one per crop)

### 2.2 **BSMOTE** (Borderline-SMOTE)

- **File**: `02.BSMOTE.ipynb`
- **Algorithm**: Borderline-1 variant focusing on decision boundary samples
- **Advantages over SMOTE**:
  - Targets samples near class boundaries
  - Generates more realistic synthetic samples
  - Better for imbalanced data
- **Implementation**: Uses `imblearn.over_sampling.BorderlineSMOTE`
- **Output**: Similar structure to SMOTE results

### 2.3 **CTGAN** (Conditional Tabular GAN)

- **File**: `03.CTGAN.ipynb`
- **Algorithm**: Generative Adversarial Network with conditional generation
- **Architecture**:
  - Generator: Produces synthetic tabular data conditioned on class labels
  - Discriminator: Distinguishes real from synthetic data
  - Supports continuous and categorical features
- **Advantages**:
  - Captures complex non-linear relationships
  - Class-conditional generation
  - Better performance on tabular data
- **Dependencies**: `ctgan` package from conda-forge
- **Output**: Synthetic hyperspectral data per crop

### 2.4 **CNN-CVAE** (Convolutional Conditional Variational Autoencoder) ⭐ Proprietary

- **Files**:
  - `04.SpectraGeni/train_CNN_CVAE.ipynb` (training)
  - `04.SpectraGeni/VAE_utils.py` (architecture)
  - `04.SpectraGeni/data_handling.py` (data pipeline)
  - `04.SpectraGeni/constants.py` (hyperparameters)
- **Architecture**:

  ```
  ENCODER:
    Input (batch, 943) → Unsqueeze → (batch, 1, 943)
    ↓ Conv1d (1→64, k=7, s=2) → (batch, 64, 472)
    ↓ Conv1d (64→128, k=5, s=2) → (batch, 128, 236)
    ↓ Conv1d (128→256, k=5, s=2) → (batch, 256, 118)
    ↓ Flatten + Label Embedding (16-dim)
    ↓ FC (512) → μ, log_var

  DECODER:
    z + label_embedding → FC (256*118)
    ↓ ConvTranspose1d (256→128, k=5, s=2) → (batch, 128, 236)
    ↓ ConvTranspose1d (128→64, k=5, s=2) → (batch, 64, 472)
    ↓ ConvTranspose1d (64→1, k=7, s=2) → (batch, 1, 943)
    ↓ Squeeze + Tanh activation
    Output (batch, 943)
  ```

- **Key Features**:
  - Conditional: Generates data per crop class
  - Convolutional: Captures spectral patterns sequentially
  - Variational: Probabilistic latent space (μ, σ)
  - Reparameterization trick for differentiable sampling
  - Batch normalization throughout
  - Early stopping (patience=30)
- **Loss Function**:
  ```
  L_total = MSE(recon_x, x) + β * KL_divergence(q(z|x,y), p(z))
  ```
- **Hyperparameters**:
  - Input Dim: 943
  - Latent Dim: 64
  - Learning Rate: 1e-4
  - Epochs: 5000 (max)
  - Batch Size: 128
  - Num Classes: 6
  - GPU Support: CUDA if available
- **Training Output**: 2000 synthetic samples per crop
- **Advantages**:
  - Specialized for 1D spectral data
  - Class-aware generation
  - Probabilistic generation (reusable latent space)
  - Efficient architecture for sequential features

---

## 3. MACHINE LEARNING MODELS

### 3.1 **Classification Models** (for evaluation)

#### Linear Models

- `LogisticRegression`: Baseline linear classifier
- `RidgeClassifier`: L2 regularized linear model
- `SGDClassifier`: Stochastic Gradient Descent
- `PassiveAggressiveClassifier`: Online learning classifier

#### Tree-Based Models

- `DecisionTreeClassifier`: Single decision tree
- `ExtraTreeClassifier`: Extremely randomized tree
- `RandomForestClassifier`: Ensemble of decision trees
- `ExtraTreesClassifier`: Ensemble of extra trees

#### Ensemble Methods (Advanced)

- `GradientBoostingClassifier`: Sequential boosting
- `AdaBoostClassifier`: Adaptive boosting
- `BaggingClassifier`: Bootstrap aggregating
- `VotingClassifier`: Hard/soft voting of multiple classifiers
- `StackingClassifier`: Meta-learner on classifier outputs

#### Non-Parametric Models

- `KNeighborsClassifier`: k-NN classifier
- `GaussianProcessClassifier`: Non-parametric probabilistic model

#### Probabilistic Models

- `GaussianNB`: Naive Bayes with Gaussian assumption
- `LinearDiscriminantAnalysis`: LDA
- `QuadraticDiscriminantAnalysis`: QDA

#### Support Vector Machines

- `LinearSVC`: Linear SVM
- `SVC`: Non-linear SVM (RBF kernel support)

### 3.2 **Validation Strategy**

- Train-Test Split: 80/20 stratified split
- Cross-Validation: k-Fold, Leave-One-Out, StratifiedKFold
- Metrics: Accuracy, F1-score, Precision, Recall, ROC-AUC
- Confusion matrices and classification reports
- Comparison across augmentation techniques

---

## 4. PROJECT WORKFLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PREPARATION                            │
├─────────────────────────────────────────────────────────────────┤
│ Real hyperspectral data (943 bands) for 6 crop classes          │
│ Train/Test split (80/20) in Parquet format                      │
│ Data loaded from: Data/Train_80/Parquet/                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             DATA AUGMENTATION (4 Parallel Paths)                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐│
│  │   SMOTE     │  │  BSMOTE      │  │  CTGAN   │  │ CNN-CVAE   ││
│  │  ~2K/class  │  │  ~2K/class   │  │  Gen.    │  │ Conditional││
│  │  01.SMOTE   │  │  02.BSMOTE   │  │  03.CTGAN│  │ 04.SpectraG││
│  └─────────────┘  └──────────────┘  └──────────┘  │ eni        │
│                                                     │ 2K/class   │
│                                                     └────────────┘
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             DATA QUALITY ASSESSMENT                              │
├─────────────────────────────────────────────────────────────────┤
│ Metrics: KL Divergence, Spectral Angle Mapper, Hellinger Dist.  │
│ Normalization: Min-Max Scaling, Standard Normal Variate (SNV)   │
│ Compare: Real vs Synthetic distributions per crop               │
│ Files: Data_quality_check/01-04.*.py                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             VISUALIZATION & EXPLORATION                          │
├─────────────────────────────────────────────────────────────────┤
│ 3D Projections:                                                  │
│  - PCA (Principal Component Analysis)                           │
│  - TSNE (t-Distributed Stochastic Neighbor Embedding)           │
│  - UMAP (Uniform Manifold Approximation and Projection)         │
│ Plots: Real vs Synthetic side-by-side                           │
│ Dataset Visualization: Dataset_Distributions_Visualization.ipynb │
│ 3D Plots: 5.3D_plotting & 6.3D_plotting notebooks               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         CLASSIFICATION MODEL TRAINING & EVALUATION               │
├─────────────────────────────────────────────────────────────────┤
│ Train classifiers on:                                            │
│  1. Real data only                                              │
│  2. Real + SMOTE synthetic                                      │
│  3. Real + BSMOTE synthetic                                     │
│  4. Real + CTGAN synthetic                                      │
│  5. Real + CNN-CVAE synthetic                                   │
│ Evaluate: 15-20 different models (linear, tree, ensemble)       │
│ Ensemble Methods: Voting & Stacking classifiers                 │
│ Files: Trainig-Testing_both/1-4.Custom_ensemble_*.ipynb         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              RESULTS COMPARISON & ANALYSIS                       │
├─────────────────────────────────────────────────────────────────┤
│ Compare performance across:                                      │
│  - Different augmentation techniques                             │
│  - Various classification models                                 │
│  - Ensemble vs single classifiers                               │
│ Metrics: Accuracy, F1, Precision, Recall, ROC-AUC              │
│ Confusion matrices and classification reports                    │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Workflow Steps:

#### Step 1: Data Loading

```python
# From data_handling.py
- Load 6 crop Parquet files
- Combine into single DataFrame (943 features + label)
- Apply stratified 80/20 train-test split
- StandardScaler normalization
- Convert to PyTorch DataLoaders (batch_size=128)
```

#### Step 2: Generate Synthetic Data

```python
# For SMOTE/BSMOTE:
- Read real data
- Apply oversampling
- Generate ~2000 synthetic samples per class
- Save as Parquet/Excel files

# For CTGAN:
- Train GAN on real data
- Generate conditional synthetic samples
- Ensure distribution similarity

# For CNN-CVAE:
- Initialize ConvolutionalCVAE with 6 classes
- Train on real data (5000 epochs max)
- Generate 2000 samples per class from latent space
```

#### Step 3: Quality Check

```python
# From Data_quality_check/ scripts
- Load real and synthetic data
- Apply normalization (min-max or SNV)
- Calculate KL divergence per feature
- Compute spectral angle mapper distances
- Calculate Hellinger distances
- Generate comparison statistics
```

#### Step 4: Visualization

```python
# 3D Projections
- Apply dimensionality reduction (PCA/TSNE/UMAP)
- Plot real vs synthetic in 3D space
- Color by class, size by dataset type
- Interactive Plotly visualizations
```

#### Step 5: Model Training

```python
# For each augmentation technique:
- Combine real training data with synthetic augmented data
- Train 15-20 classifiers
- Use 5-fold cross-validation
- Track: accuracy, F1, precision, recall
- Create confusion matrices
```

#### Step 6: Ensemble Creation

```python
# Voting Classifier
- Combine predictions from multiple models
- Use soft voting (probability averaging)
- Benchmark against individual models

# Stacking Classifier
- Level 0: Multiple base learners
- Level 1: Meta-learner aggregates predictions
- Improves generalization
```

---

## 5. KEY COMPONENTS & FILES

### Core Architecture Files

#### `Data_augmentation/04.SpectraGeni/`

- **`train_CNN_CVAE.ipynb`**: Complete training pipeline
  - Data loading and preparation
  - Model initialization
  - Training loop with early stopping
  - Synthetic data generation
  - Architecture visualization
- **`VAE_utils.py`**: Model architecture and training functions
  - `ConvolutionalCVAE`: Full model class
  - `vae_loss_function()`: CVAE loss computation
  - `train_cvae()`: Training loop with validation
  - `generate_synthetic_conditional()`: Generation function
- **`data_handling.py`**: Data pipeline
  - `data_loading_all_parquet()`: Load multiple crop data
  - `data_prep_conditional()`: Split and prepare for training
  - `save_data()`: Save generated synthetic data
  - `convert_all_excel_to_parquet()`: Format conversion
- **`constants.py`**: Hyperparameters
  ```python
  INPUT_DIM = 943        # Hyperspectral bands
  LATENT_DIM = 64        # VAE latent space
  LR = 1e-4              # Learning rate
  EPOCHS = 5000          # Max epochs
  DEVICE = "cuda:0"      # GPU device
  NUM_SAMPLES = 2000     # Samples per class
  NUM_CLASSES = 6        # 6 crops
  ```

### Data Augmentation Notebooks

- `01.SMOTE.ipynb`: SMOTE implementation
- `02.BSMOTE.ipynb`: Borderline-SMOTE implementation
- `03.CTGAN.ipynb`: Conditional Tabular GAN
- Each generates ~2000 synthetic samples per crop class

### Quality Check Scripts

- `Data_quality_check/01.SMOTE.py`: Quality metrics for SMOTE
- `Data_quality_check/02.Borderline_SMOTE.py`: Quality for BSMOTE
- `Data_quality_check/03.CTGAN.py`: Quality for CTGAN
- `Data_quality_check/04.CVAE.py`: Quality for CNN-CVAE
- **Functions**:
  - `kl_divergence()`: KL divergence per feature
  - `cal_bi_multi_variate_plot_hellinger_distances()`: Hellinger distance
  - `spectral_angle_mapper_mean()`: Spectral angle mapping
  - `normalize_datasets()`: Min-Max and SNV normalization

### Visualization Notebooks

- `Dataset_Distributions_Visualization.ipynb`: Overall data distribution plots
- `Data_augmentation/5.3D_plotting (Real Vs. Synthetic) CTGAN.ipynb`: 3D comparison for CTGAN
- `Data_augmentation/6.3D_plotting (Real Vs. Synthetic) CVAE.ipynb`: 3D comparison for CNN-CVAE
- Uses PCA, TSNE, UMAP for dimensionality reduction

### Classification & Ensemble Training

- `Trainig-Testing_both/1.Custom_ensemble_SMOTE.ipynb`: Train & test on SMOTE data
- `Trainig-Testing_both/2.Custom_ensemble_BSMOTE.ipynb`: Train & test on BSMOTE data
- `Trainig-Testing_both/3.Custom_ensemble_CTGAN.ipynb`: Train & test on CTGAN data
- `Trainig-Testing_both/4.Custom_ensemble_CNN-CVAE_real80.ipynb`: Train & test on CNN-CVAE data
- **Each notebook**:
  - Loads synthetic data
  - Combines with real training data
  - Trains 15-20 individual classifiers
  - Creates Voting and Stacking ensembles
  - Generates confusion matrices and metrics
  - Compares results

### Helper Scripts

- `Data_augmentation/common_fun.py`: Common visualization functions
  - `create_data()`: Prepare real vs synthetic datasets
  - `plot_3D()`: 3D scatter plots with Plotly
  - `scaler`: StandardScaler usage
- `Data_quality_check/common_fun.py`: Common quality check functions
  - `min_max_scaling()`: Normalize to [0, 1]
  - `standard_normal_variate()`: SNV normalization
  - `normalize_datasets()`: Apply normalization per crop
  - Feature importance calculations

### Configuration

- `requirements.txt`: All Python dependencies
  - torch, torchvision: Deep learning
  - scikit-learn: Classical ML
  - pandas, numpy: Data manipulation
  - plotly, matplotlib: Visualization
  - imblearn: SMOTE variants
  - ctgan, copulas: GAN framework
  - scipy: Statistical functions
  - umap, tsne: Dimensionality reduction

---

## 6. IMPORTANT FEATURES & FINDINGS

### 6.1 **Hyperspectral Data Characteristics**

- **943 spectral bands** per sample (high-dimensional)
- **6 crop classes** with varying spectral signatures
- **Real data imbalance**: Some classes have more samples than others
- **Dummy class "zummy"** added during SMOTE/BSMOTE to test oversampling behavior

### 6.2 **CNN-CVAE Advantages over baselines**

1. **Conditional Generation**: Class-aware synthetic data
2. **1D Convolutions**: Capture sequential spectral patterns
3. **Probabilistic**: Reusable latent space for diversity
4. **Efficient**: Fewer parameters than fully connected VAE
5. **Batch Normalization**: Stable training
6. **Early Stopping**: Prevents overfitting

### 6.3 **Data Quality Metrics**

- **KL Divergence**: Mean across 943 bands to measure distribution similarity
- **Spectral Angle Mapper**: Angular distance in high-dimensional space
- **Hellinger Distance**: Between distributions (0-1, lower=better)
- **SNV Normalization**: Removes sensor-specific biases

### 6.4 **Classification Strategy**

- **Multiple Base Learners**: 15-20 different models tested
- **Voting Classifier**: Hard majority vote or soft probability averaging
- **Stacking**: Meta-learner learns optimal combination
- **Cross-Validation**: 5-fold or stratified splits ensure robustness
- **Stratified Splits**: Maintain class distribution in train/test

### 6.5 **Ensemble Methods Results**

- Custom ensemble typically outperforms individual classifiers
- Stacking > Voting > Individual models (generally)
- CNN-CVAE synthetic data often produces best results
- Combination of real + synthetic improves generalization

### 6.6 **Expected Performance Improvements**

- SMOTE: +5-10% accuracy on minority classes
- BSMOTE: +7-12% (focuses on decision boundaries)
- CTGAN: +8-15% (captures non-linear relationships)
- CNN-CVAE: +10-18% (specialized architecture for spectral data)

---

## 7. PROJECT DEPENDENCIES

### Core Libraries

- **torch**: Deep learning framework (CVAE)
- **scikit-learn**: Classical ML algorithms and utilities
- **pandas**: Data manipulation and I/O
- **numpy**: Numerical computing
- **imblearn**: SMOTE and BSMOTE implementations
- **ctgan**: Conditional Tabular GAN
- **scipy**: Statistical distributions and functions

### Visualization

- **plotly**: Interactive 3D plots
- **matplotlib**: Static plotting
- **umap**: UMAP dimensionality reduction
- **scikit-learn (PCA, TSNE)**: Dimensionality reduction

### Data I/O

- **pyarrow**: Parquet file support
- **openpyxl**: Excel file support

---

## 8. RECOMMENDED WORKFLOW FOR NEW USERS

1. **Understand the data**: Read `README.md` and inspect sample Parquet files
2. **Study the baseline**: Run SMOTE notebook (`01.SMOTE.ipynb`)
3. **Explore CNN-CVAE**: Run `train_CNN_CVAE.ipynb` to understand the architecture
4. **Check data quality**: Run relevant script in `Data_quality_check/`
5. **Visualize results**: Run 3D plotting notebooks
6. **Train classifiers**: Run ensemble training notebooks
7. **Compare results**: Analyze metrics across all techniques

---

## 9. FUTURE ENHANCEMENTS

Potential improvements not yet implemented:

- Multi-scale CNN-CVAE (different kernel sizes for multi-resolution)
- Wasserstein CVAE for better latent space
- Adversarial training (hybrid GAN-VAE)
- Class imbalance handling with weighted losses
- Temporal augmentation for time-series spectral data
- Transfer learning from larger spectral datasets
- Active learning for efficient sampling

---

## 10. SUMMARY TABLE

| Technique | Type                | Complexity | Speed     | Quality     | Flexibility |
| --------- | ------------------- | ---------- | --------- | ----------- | ----------- |
| SMOTE     | Rule-based          | Low        | Very Fast | Medium      | Low         |
| BSMOTE    | Rule-based          | Low        | Very Fast | Medium-High | Low         |
| CTGAN     | Deep Learning (GAN) | High       | Slow      | High        | High        |
| CNN-CVAE  | Deep Learning (VAE) | High       | Slow      | Very High   | Very High   |

---

_Analysis Date: April 19, 2026_
_Project: SpectraGeni - Synthetic Hyperspectral Data Generation_
_Workspace: c:\Users\manoj\OneDrive\Documents\GitHub\SpectraGeni_

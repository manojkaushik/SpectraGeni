# SpectraGeni

**Advanced Synthetic Hyperspectral Data Generation for Crop Classification**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Overview

SpectraGeni is a comprehensive framework for generating high-quality synthetic hyperspectral data for agricultural crop classification. The project implements and compares four advanced data augmentation techniques—SMOTE, Borderline-SMOTE (BSMOTE), Conditional Tabular GAN (CTGAN), and a proprietary Convolutional Conditional Variational Autoencoder (CNN-CVAE)—to address class imbalance and expand training datasets for hyperspectral crop classification tasks.

The project evaluates machine learning classifiers and employs ensemble learning strategies to achieve robust crop classification with synthetic data.

## Key Features

- **Multiple Data Augmentation Techniques**: SMOTE, BSMOTE, CTGAN, and proprietary CNN-CVAE
- **943-Dimensional Hyperspectral Data**: Real-world agricultural spectral data with 6 crop classes
- **Advanced Quality Metrics**: Hellinger distance, Pairwise Correlation Difference, Propensity Score, Spectral Angle Mapper (SAM)
- **Comprehensive ML Pipeline**: ML classifiers with custom ensemble methods (Voting, Stacking)
- **Data Visualization**: 3D projection comparisons (PCA, UMAP, t-SNE) of real vs. synthetic data
- **Production-Ready**: Modular code with common utilities for data handling and augmentation

## Dataset

- **Crop Classes**: 6 agricultural crop varieties (Capsicum, Chilli, Mulberry, Potato, Tomato_Healthy, Tomato_Unhealthy)
- **Features**: 943-dimensional hyperspectral bands (reflectance spectra)
- **Techniques**: Augmentation from ~200-300 real samples to 2000-5000 synthetic samples per class
- **Storage Format**: CSV/DataFrame-compatible structure

## Project Structure

```
SpectraGeni/
├── README.md                                   # Project documentation
├── LICENSE                                     # MIT License
├── requirements.txt                            # Python dependencies
├── Dataset_Distributions_Visualization.ipynb   # Visual data exploration
│
├── Data_augmentation/                          # Data augmentation techniques
│   ├── 01.SMOTE.ipynb                         # SMOTE implementation
│   ├── 02.BSMOTE.ipynb                        # Borderline-SMOTE
│   ├── 03.CTGAN.ipynb                         # Conditional Tabular GAN
│   ├── 5.3D_plotting (Real Vs. Synthetic) CTGAN.ipynb
│   ├── 6.3D_plotting (Real Vs. Synthetic) CVAE.ipynb
│   ├── common_fun.py                          # Shared utilities
│   ├── KLdivergence.py                        # Quality metrics
│   └── 04.SpectraGeni/                        # CNN-CVAE (Proprietary)
│       ├── constants.py                       # Configuration constants
│       ├── data_handling.py                   # Data loading/preprocessing
│       ├── VAE_utils.py                       # CVAE architecture & training
│       └── train_CNN_CVAE.ipynb               # Main CVAE training pipeline
│
├── Data_quality_check/                        # Data quality evaluation
│   ├── 01.SMOTE.py                           # SMOTE quality assessment
│   ├── 02.Borderline_SMOTE.py                # BSMOTE quality assessment
│   ├── 03.CTGAN.py                           # CTGAN quality assessment
│   ├── 04.CVAE.py                            # CVAE quality assessment
│   └── common_fun.py                          # Quality metric utilities
│
└── Training-Testing_both/                    # Classification & Ensembles
    ├── 1.Custom_ensemble_SMOTE.ipynb          # SMOTE + ensemble classifier
    ├── 2.Custom_ensemble_BSMOTE.ipynb         # BSMOTE + ensemble classifier
    ├── 3.Custom_ensemble_CTGAN.ipynb          # CTGAN + ensemble classifier
    └── 4.Custom_ensemble_CNN-CVAE_real80.ipynb # CNN-CVAE + ensemble (80% real data)
```

## Data Augmentation Techniques

### 1. **SMOTE (Synthetic Minority Over-sampling Technique)**

- Classical k-NN based oversampling algorithm
- Generates synthetic samples along line segments between k-nearest neighbors
- Output: ~2,000 samples per crop class
- Best for: Balanced, linear data distributions

### 2. **BSMOTE (Borderline-SMOTE)**

- Variant focusing on borderline samples (danger zone)
- Only generates synthetic samples near decision boundaries
- Output: ~2,000 samples per crop class
- Best for: Improving classifier generalization with targeted synthesis

### 3. **CTGAN (Conditional Tabular GAN)**

- Generative Adversarial Network for high-dimensional tabular data
- Conditional generation based on crop class
- Output: ~2,000 samples per crop class
- Best for: Learning complex, non-linear distributions

### 4. **CNN-CVAE (Convolutional Conditional Variational Autoencoder)** ⭐

- Proprietary deep learning architecture
- Custom 1D CNN encoder/decoder (943 → 64 latent → 943)
- Conditional generation on crop class labels
- Training: 5,000 epochs with early stopping
- Output: ~5,000 samples per crop class
- Best for: High-fidelity spectral data synthesis with regularization

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- 8GB+ RAM (16GB+ recommended for CTGAN/CVAE training)
- GPU support (optional but recommended for deep learning models)

### Step 1: Clone the Repository

```bash
git clone https://github.com/manojkaushik/SpectraGeni.git
cd SpectraGeni
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n spectrageni python=3.8
conda activate spectrageni
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow, ctgan, pandas, sklearn; print('All packages installed successfully!')"
```

## Usage Guide

### Workflow Overview

```
1. Data Loading & Exploration
   └─→ Dataset_Distributions_Visualization.ipynb

2. Data Augmentation (Choose one or multiple techniques)
   ├─→ Data_augmentation/01.SMOTE.ipynb
   ├─→ Data_augmentation/02.BSMOTE.ipynb
   ├─→ Data_augmentation/03.CTGAN.ipynb
   └─→ Data_augmentation/04.SpectraGeni/train_CNN_CVAE.ipynb

3. Data Quality Assessment
   ├─→ Data_quality_check/01.SMOTE.py
   ├─→ Data_quality_check/02.Borderline_SMOTE.py
   ├─→ Data_quality_check/03.CTGAN.py
   └─→ Data_quality_check/04.CVAE.py

4. 3D Visualization (Real vs. Synthetic)
   ├─→ Data_augmentation/5.3D_plotting (Real Vs. Synthetic) CTGAN.ipynb
   └─→ Data_augmentation/6.3D_plotting (Real Vs. Synthetic) CVAE.ipynb

5. Classification & Ensemble Learning
   ├─→ Training-Testing_both/1.Custom_ensemble_SMOTE.ipynb
   ├─→ Training-Testing_both/2.Custom_ensemble_BSMOTE.ipynb
   ├─→ Training-Testing_both/3.Custom_ensemble_CTGAN.ipynb
   └─→ Training-Testing_both/4.Custom_ensemble_CNN-CVAE_real80.ipynb
```

## Dependencies

Key packages (see `requirements.txt` for complete list):

- **Data Science**: pandas, numpy, scikit-learn, scipy
- **Deep Learning**: tensorflow, keras, pytorch (for CVAE)
- **Augmentation**: imbalanced-learn, ctgan
- **Visualization**: matplotlib, plotly, seaborn, umap, scikit-learn
- **Utilities**: Faker, graphviz, jupyter

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure code follows PEP 8 style guidelines and includes documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hyperspectral crop dataset from agricultural research sources
- SMOTE implementation: [imbalanced-learn](https://imbalanced-learn.org/)
- CTGAN implementation: [CTGAN](https://github.com/sdv-dev/CTGAN)
- Visualization tools: Plotly, Matplotlib, UMAP, t-SNE

## Troubleshooting

### Issue: Out of Memory Error during CTGAN/CVAE training

**Solution**: Reduce batch size in constants.py or use data sampling

### Issue: Poor synthetic data quality

**Solution**: Try different augmentation techniques or adjust hyperparameters

### Issue: GPU not detected

**Solution**: Install GPU support for TensorFlow/PyTorch (CUDA, cuDNN)

### Issue: Missing dependencies

**Solution**: Run `pip install -r requirements.txt` again with `--upgrade` flag

## Future Enhancements

- [ ] Multi-class imbalance handling for severely skewed datasets
- [ ] Transfer learning from pre-trained models
- [ ] Real-time synthetic data generation API
- [ ] Interactive web dashboard for data exploration
- [ ] Extended evaluation on multiple hyperspectral datasets
- [ ] Federated learning support for distributed training

## Author

**Manoj Kaushik**

- GitHub: [@manojkaushik](https://github.com/manojkaushik)
- Email: manojkaushik93@gmail.com

---

**Status**: Active Development | **Last Updated**: April 2026 | **Version**: 1.0.0

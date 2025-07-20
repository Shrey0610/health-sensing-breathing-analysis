# Health Sensing Breathing Analysis - DeepMedico™ 🏥

A comprehensive machine learning project for detecting breathing irregularities during sleep using physiological signals from overnight sleep studies.

## 📋 Project Overview

DeepMedico™ has collected overnight sleep data (8 hours per participant) from 5 subjects (AP01-AP05) to develop automated breathing irregularity detection systems. This project implements the complete pipeline from data visualization to machine learning model training.

### 🔬 Available Data
For each participant, the following physiological signals are available:
- **Nasal Airflow** (sampled at 32 Hz)
- **Thoracic Movement** (sampled at 32 Hz)
- **SpO₂ (Oxygen Saturation)** (sampled at 4 Hz)
- **Event annotations** for breathing irregularities (Hypopnea, Obstructive Apnea)
- **Sleep profile** recordings for sleep stage analysis

## 📊 Task 1: Data Visualization ✅ COMPLETED

Created comprehensive visualizations of physiological signals across 8-hour sleep sessions for all participants.

### 🎯 Implementation
- **Script**: [`scripts/vis.ipynb`](scripts/vis.ipynb)
- **Output**: [`Visualizations/`](Visualizations/) directory

### 📈 Generated Visualizations
- [AP01_visualization.pdf](Visualizations/AP01_visualization.pdf)
- [AP02_visualization.pdf](Visualizations/AP02_visualization.pdf) 
- [AP03_visualization.pdf](Visualizations/AP03_visualization.pdf)
- [AP04_visualization.pdf](Visualizations/AP04_visualization.pdf)
- [AP05_visualization.pdf](Visualizations/AP05_visualization.pdf)

### 🔧 Features
- Multi-panel plots showing Nasal Airflow, Thoracic Movement, and SpO₂ signals
- Overlay of breathing event annotations (Hypopnea, Obstructive Apnea)
- 8-hour timeline visualization
- PDF export format as requested

**Usage Example:**
```bash
python vis.py -name "Data/AP01"
```

## 🧹 Task 2: Data Cleaning ✅ COMPLETED

Implemented signal preprocessing to filter high-frequency noise while preserving breathing patterns.

### 🎯 Implementation Details
- **Filtering**: Applied low-pass filters considering human breathing rate (0.17-0.4 Hz)
- **Noise Reduction**: Removed high-frequency artifacts while maintaining signal integrity
- **Signal Quality**: Improved data quality for downstream ML analysis

## 🏗️ Task 3: Dataset Creation ✅ COMPLETED

Created a comprehensive ML-ready dataset from preprocessed physiological signals.

### 🎯 Implementation
- **Script**: [`scripts/create_dataset.ipynb`](scripts/create_dataset.ipynb)
- **Class**: `DatasetCreator` with comprehensive processing pipeline

### 📊 Dataset Specifications
- **Window Size**: 30-second segments with 50% overlap (15-second step)
- **Labeling Strategy**: Based on >50% overlap with breathing events
- **Target Labels**: Hypopnea, Obstructive Apnea, Normal
- **Total Windows**: 8,674 labeled windows across all participants

### 📁 Generated Files
- [`Dataset/breathing_dataset_features.parquet`](Dataset/breathing_dataset_features.parquet) - ML-ready features
- [`Dataset/breathing_dataset_features.csv`](Dataset/breathing_dataset_features.csv) - Human-readable format
- [`Dataset/dataset_stats.json`](Dataset/dataset_stats.json) - Comprehensive statistics

### 📈 Dataset Statistics
```json
{
  "total_windows": 8674,
  "participants": 5,
  "window_duration_seconds": 30,
  "overlap_ratio": 0.5,
  "overall_distribution": {
    "Normal": 7841,
    "Hypopnea": 775,
    "Obstructive Apnea": 58
  }
}
```

### 👥 Per-Participant Distribution
- **AP01**: 1,575 windows (Normal: 1,406, Hypopnea: 165, Obstructive Apnea: 4)
- **AP02**: 1,769 windows (Normal: 1,616, Hypopnea: 150, Obstructive Apnea: 3)
- **AP03**: 1,696 windows (Normal: 1,679, Hypopnea: 16, Obstructive Apnea: 1)
- **AP04**: 1,932 windows (Normal: 1,765, Hypopnea: 166, Obstructive Apnea: 1)
- **AP05**: 1,702 windows (Normal: 1,375, Hypopnea: 278, Obstructive Apnea: 49)

**Usage Example:**
```bash
python create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

## 🤖 Task 4: Machine Learning Models ✅ COMPLETED

Implemented and trained deep learning models for breathing irregularity classification.

### 🎯 Model Architectures
- **1D Convolutional Neural Network (1D CNN)**
- **1D Conv-LSTM** for temporal pattern recognition

### 📊 Training Configuration
- **Cross-Validation**: Leave-One-Participant-Out (LOPO)
- **Evaluation Metrics**: Accuracy, Precision, Recall, Sensitivity, Specificity
- **Output**: Comprehensive confusion matrices and performance reports

### 💾 Model Artifacts
- [`models/label_encoder_20250713_194527.pkl`](models/label_encoder_20250713_194527.pkl)
- [`models/training_results_20250713_194527.json`](models/training_results_20250713_194527.json)
- [`models/load_models_example_20250713_194527.py`](models/load_models_example_20250713_194527.py)

### 📈 Implementation
- **Script**: [`scripts/train_model.ipynb`](scripts/train_model.ipynb)
- **Validation Strategy**: Robust LOPO cross-validation ensuring no data leakage
- **Class Balance**: Addressed class imbalance with appropriate techniques

## 🎁 Bonus: Sleep Stage Classification

Extended analysis for sleep stage classification using the same modeling approaches to predict Wake, REM, N1, N2, and N3 stages.

## 🚀 Getting Started

### 📋 Prerequisites
```bash
pip install -r requirements.txt
```

### 🔄 Workflow
1. **Visualize Data**: Run visualization notebook to explore signals
2. **Create Dataset**: Execute dataset creation to generate ML-ready data
3. **Train Models**: Run model training with cross-validation
4. **Evaluate Results**: Analyze performance metrics and confusion matrices

### 📊 Project Structure
```
health-sensing-breathing-analysis/
├── Data/                          # Raw participant data (AP01-AP05)
├── Dataset/                       # Processed ML-ready datasets
├── models/                        # Trained models and artifacts
├── scripts/                       # Jupyter notebooks for analysis
├── Visualizations/                # Generated PDF visualizations
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 📊 Key Results

### Dataset Quality
- ✅ **8,674 labeled windows** extracted from 5 participants
- ✅ **Balanced temporal coverage** with 50% overlap strategy
- ✅ **Multi-modal signals** (Airflow, Thoracic Movement, SpO₂)
- ✅ **Event-based labeling** with >50% overlap threshold

### Model Performance
- ✅ **Robust cross-validation** with Leave-One-Participant-Out
- ✅ **Deep learning architectures** optimized for time-series classification
- ✅ **Comprehensive evaluation** with multiple performance metrics

## 🏷️ Labels Distribution
The dataset contains three main classes with the following distribution:
- **Normal**: 90.4% (7,841 windows)
- **Hypopnea**: 8.9% (775 windows)  
- **Obstructive Apnea**: 0.7% (58 windows)

## 🔬 Technical Highlights
- **Real-time Processing**: 30-second window analysis suitable for clinical deployment
- **Multi-signal Fusion**: Leveraging complementary physiological signals
- **Temporal Modeling**: Conv-LSTM captures breathing pattern dynamics
- **Clinical Relevance**: Focus on clinically significant breathing irregularities

---

**🏥 DeepMedico™ - Advancing Healthcare Through AI**

*This project demonstrates end-to-end development of a breathing irregularity detection system using machine learning on overnight sleep study data.*
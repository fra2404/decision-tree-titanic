# 🚢 Titanic Survival Prediction using Optimized Decision Tree

**BUAA Artificial Intelligence Assignment 2**  
**Author:** Francesco Albano  
**Student ID:** lj2506219  
**Date:** October 2025

## 🎯 Project Overview

This project implements an optimized decision tree algorithm to predict Titanic passenger survival using advanced feature engineering, intelligent feature selection, and systematic hyperparameter optimization.

### Key Achievements

- **75.37% validation accuracy** with optimized decision tree
- **Advanced feature engineering** creating 17 features from original 11
- **Intelligent feature selection** reducing features by 47% while improving performance
- **Professional ML workflow** with comprehensive evaluation and visualization

## 📁 Project Structure

```
Assignment-02/
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   └── gender_submission.csv  # Sample submission format
├── src/
│   └── titanic_decision_tree.py      # Optimized implementation (75.37% accuracy)
├── results/
│   ├── eda_plots.png                 # Exploratory data analysis plots
│   ├── correlation_matrix.png        # Feature correlation heatmap
│   ├── model_evaluation.png          # Model performance visualization
│   ├── decision_tree_full.png        # Complete decision tree visualization
│   └── titanic_predictions.csv       # Final predictions
├── Lab_Report_Titanic_Decision_Tree.md  # Comprehensive lab report
├── README.md                         # This file
├── Assignment-02.pdf                 # Assignment requirements
├── create_submission_package.sh      # Automated packaging script
└── requirements.txt                  # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
# Run the optimized decision tree analysis
cd src
python titanic_decision_tree.py
```

### 3. Create Submission Package

```bash
# Create a complete submission package
./create_submission_package.sh
```

## 🔍 Key Features

### Algorithm Implementation

- **Optimized Decision Tree**: Professional scikit-learn implementation
- **Advanced Feature Engineering**: 17 features created, 9 optimally selected
- **Hyperparameter Tuning**: GridSearchCV with 1,500 parameter combinations
- **Performance**: 75.37% validation accuracy with controlled overfitting

### Data Preprocessing

- **Intelligent Missing Value Imputation**: Median by title/class for Age
- **Advanced Feature Engineering**: Fare_Per_Person, Age_Class interactions
- **Feature Selection**: Systematic importance-based optimization
- **Categorical Encoding**: Label encoding optimized for decision trees

### Feature Engineering

- **Title extraction**: Mr, Mrs, Miss, Master, and Rare categories
- **Economic features**: Fare_Per_Person (20.69% importance)
- **Interaction features**: Age_Class combinations
- **Family dynamics**: FamilySize optimization
- **9 final features**: Intelligently selected from 17 engineered features

## 📊 Results Summary

### Model Performance

- **Validation Accuracy**: 75.37%
- **Cross-validation**: 84.41% ± 8.51%
- **Training Accuracy**: 87.98%
- **Overfitting Gap**: 12.61% (well-controlled)
- **Feature Importance**: Sex (50.07%), Fare_Per_Person (20.69%), FamilySize (10.63%)

### Key Insights

1. **Gender dominates**: "Women and children first" policy clearly visible
2. **Class matters**: Socioeconomic status significantly affects survival
3. **Age factor**: Children have higher survival rates regardless of gender
4. **Family dynamics**: Complex relationship with survival chances

### Confusion Matrix (Validation Set)

|                     | Predicted Died | Predicted Survived |
| ------------------- | -------------- | ------------------ |
| **Actual Died**     | 70             | 13                 |
| **Actual Survived** | 20             | 31                 |

**Metrics**: Precision 75%, Recall 75%, F1-Score 75%

## 🛠 Technical Details

### Hyperparameter Optimization

- **Max Depth**: 7 (optimal complexity-performance balance)
- **Min Samples Split**: 10 (overfitting prevention)
- **Min Samples Leaf**: 2 (leaf granularity)
- **Max Features**: 'sqrt' (feature sampling for generalization)
- **Criterion**: Gini (optimal for this dataset)

### Validation Strategy

- **Train/Validation Split**: 85/15 stratified
- **Cross-validation**: 10-fold stratified
- **Search Space**: 1,500 parameter combinations
- **Metrics**: Accuracy, Precision, Recall, F1-score, Overfitting analysis

## 📈 Visualizations

The project generates several visualizations:

1. **EDA Plots**: Survival distributions by various features
2. **Correlation Matrix**: Feature relationships heatmap
3. **Model Evaluation**: Confusion matrix, feature importance, tree structure
4. **Decision Tree**: Complete tree visualization (depth=7)

## 🎓 Educational Value

### Skills Demonstrated

- **Professional ML Implementation**: Scikit-learn with advanced optimization
- **Feature Engineering Mastery**: Creative feature creation and intelligent selection
- **Hyperparameter Optimization**: Systematic GridSearchCV with cross-validation
- **Model Evaluation**: Comprehensive performance assessment and overfitting control
- **Data Science Pipeline**: Complete workflow from EDA to production-ready model
- **Visualization**: Professional presentation of results and insights

### Learning Outcomes

- Deep understanding of decision tree mechanics
- Practical experience with data preprocessing
- Hyperparameter optimization techniques
- Model interpretation and business insight extraction

## 📖 Documentation

For detailed analysis, methodology, and results interpretation, see:

- **[Lab Report](Lab_Report_Titanic_Decision_Tree.md)**: Comprehensive documentation

## 🔧 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- graphviz (for tree visualization)

## 👨‍💻 Author

**Francesco Albano**  
BUAA Artificial Intelligence Course  
October 2025

## 📄 License

This project is created for educational purposes as part of university coursework.

---

**Note**: This implementation demonstrates professional-grade machine learning methodology with advanced feature engineering, systematic optimization, and production-ready code quality suitable for real-world data science applications.

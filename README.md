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
│   ├── titanic_decision_tree.py      # Main implementation (scikit-learn)
│   └── decision_tree_from_scratch.py # From-scratch implementation
├── results/
│   ├── eda_plots.png                 # Exploratory data analysis plots
│   ├── correlation_matrix.png        # Feature correlation heatmap
│   ├── model_evaluation.png          # Model performance visualization
│   ├── decision_tree_full.png        # Complete decision tree visualization
│   └── titanic_predictions.csv       # Final predictions
├── Lab_Report_Titanic_Decision_Tree.md  # Comprehensive lab report
├── README.md                         # This file
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
# Run the main analysis (scikit-learn implementation)
cd src
python titanic_decision_tree.py

# Run the from-scratch implementation
python decision_tree_from_scratch.py
```

## 🔍 Key Features

### Algorithm Implementation

- **From-scratch decision tree**: Custom implementation using CART algorithm
- **Scikit-learn optimization**: Hyperparameter tuning with GridSearchCV
- **Performance comparison**: Both implementations achieve ~75-77% accuracy

### Data Preprocessing

- **Missing value imputation**: Smart strategies for Age, Embarked, and Fare
- **Feature engineering**: Title extraction, family size, age/fare groupings
- **Categorical encoding**: Label encoding for tree-compatible format

### Feature Engineering

- **Title extraction**: Mr, Mrs, Miss, Master, and Rare categories
- **Family dynamics**: FamilySize and IsAlone features
- **Categorical binning**: Age groups and fare groups
- **12 final features**: Optimally selected for decision tree performance

## 📊 Results Summary

### Model Performance

- **Validation Accuracy**: 76.54%
- **Cross-validation**: 81.47% ± 4.87%
- **Feature Importance**: Sex (41.4%), Pclass (19.9%), Fare (14.4%)

### Key Insights

1. **Gender dominates**: "Women and children first" policy clearly visible
2. **Class matters**: Socioeconomic status significantly affects survival
3. **Age factor**: Children have higher survival rates regardless of gender
4. **Family dynamics**: Complex relationship with survival chances

### Confusion Matrix (Validation Set)

|                     | Predicted Died | Predicted Survived |
| ------------------- | -------------- | ------------------ |
| **Actual Died**     | 96             | 14                 |
| **Actual Survived** | 28             | 41                 |

## 🛠 Technical Details

### Decision Tree Algorithm (From Scratch)

- **Splitting Criterion**: Gini impurity
- **Tree Construction**: Recursive binary splitting
- **Stopping Criteria**: Max depth, min samples, purity
- **Implementation**: Object-oriented design with Node class

### Hyperparameter Optimization

- **Max Depth**: 7 (optimal)
- **Min Samples Split**: 2
- **Min Samples Leaf**: 5
- **Criterion**: Entropy (outperformed Gini)

### Validation Strategy

- **Train/Validation Split**: 80/20 stratified
- **Cross-validation**: 5-fold stratified
- **Metrics**: Accuracy, Precision, Recall, F1-score

## 📈 Visualizations

The project generates several visualizations:

1. **EDA Plots**: Survival distributions by various features
2. **Correlation Matrix**: Feature relationships heatmap
3. **Model Evaluation**: Confusion matrix, feature importance, tree structure
4. **Decision Tree**: Complete tree visualization (depth=7)

## 🎓 Educational Value

### Skills Demonstrated

- **Algorithm Implementation**: From-scratch decision tree using CART
- **Data Science Pipeline**: Complete ML workflow from EDA to deployment
- **Feature Engineering**: Creative feature extraction and selection
- **Model Evaluation**: Proper validation and performance assessment
- **Visualization**: Clear presentation of results and insights

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

**Note**: This implementation demonstrates both theoretical understanding (from-scratch) and practical application (scikit-learn) of decision tree algorithms for real-world data science problems.

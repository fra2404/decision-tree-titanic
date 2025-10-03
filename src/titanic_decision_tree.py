#!/usr/bin/env python3
"""
Titanic Survival Prediction using Optimized Decision Tree Algorithm
BUAA Artificial Intelligence Assignment 2

This script implements an optimized decision tree algorithm to predict Titanic survival.
It includes advanced feature engineering, intelligent feature selection, data preprocessing, 
model training with hyperparameter tuning, evaluation, and comprehensive visualization.

Key Optimizations:
- Feature selection based on importance analysis
- Overfitting reduction through careful feature curation
- Advanced preprocessing and feature engineering
- Comprehensive hyperparameter tuning

Author: Francesco Albano - lj2506219
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TitanicDecisionTree:
    """
    A class to implement and analyze decision tree for Titanic survival prediction.
    """
    
    def __init__(self, data_path="../data/"):
        """
        Initialize the TitanicDecisionTree class.
        
        Args:
            data_path (str): Path to the data directory
        """
        self.data_path = data_path
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None
        
    def load_data(self):
        """Load the Titanic dataset."""
        print("Loading Titanic dataset...")
        try:
            self.train_data = pd.read_csv(f"{self.data_path}train.csv")
            self.test_data = pd.read_csv(f"{self.data_path}test.csv")
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def exploratory_data_analysis(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic information about the dataset
        print("\nDataset Info:")
        print(self.train_data.info())
        
        print("\nFirst 5 rows of training data:")
        print(self.train_data.head())
        
        print("\nBasic statistics:")
        print(self.train_data.describe())
        
        print("\nSurvival rate:")
        survival_rate = self.train_data['Survived'].mean()
        print(f"Overall survival rate: {survival_rate:.2%}")
        
        # Missing values analysis
        print("\nMissing values:")
        missing_values = self.train_data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Create visualizations
        self._create_eda_plots()
        
    def _create_eda_plots(self):
        """Create exploratory data analysis plots."""
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Titanic Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Survival count
        survival_counts = self.train_data['Survived'].value_counts()
        axes[0, 0].bar(['Died', 'Survived'], survival_counts.values, color=['red', 'green'], alpha=0.7)
        axes[0, 0].set_title('Survival Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Survival by gender
        survival_by_sex = pd.crosstab(self.train_data['Sex'], self.train_data['Survived'])
        survival_by_sex.plot(kind='bar', ax=axes[0, 1], color=['red', 'green'], alpha=0.7)
        axes[0, 1].set_title('Survival by Gender')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(['Died', 'Survived'])
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # 3. Survival by passenger class
        survival_by_class = pd.crosstab(self.train_data['Pclass'], self.train_data['Survived'])
        survival_by_class.plot(kind='bar', ax=axes[0, 2], color=['red', 'green'], alpha=0.7)
        axes[0, 2].set_title('Survival by Passenger Class')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend(['Died', 'Survived'])
        axes[0, 2].tick_params(axis='x', rotation=0)
        
        # 4. Age distribution
        axes[1, 0].hist(self.train_data['Age'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Age Distribution')
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Fare distribution
        axes[1, 1].hist(self.train_data['Fare'].dropna(), bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('Fare Distribution')
        axes[1, 1].set_xlabel('Fare')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. Survival by embarked port
        survival_by_embarked = pd.crosstab(self.train_data['Embarked'], self.train_data['Survived'])
        survival_by_embarked.plot(kind='bar', ax=axes[1, 2], color=['red', 'green'], alpha=0.7)
        axes[1, 2].set_title('Survival by Embarked Port')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].legend(['Died', 'Survived'])
        axes[1, 2].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('../results/eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation matrix
        plt.figure(figsize=(10, 8))
        # Select only numeric columns for correlation
        numeric_columns = self.train_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.train_data[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('../results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for decision tree training."""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Make copies to avoid modifying original data
        train_processed = self.train_data.copy()
        test_processed = self.test_data.copy()
        
        # Feature Engineering
        print("\nFeature Engineering:")
        
        # 1. Extract titles from names
        def extract_title(name):
            return name.split(',')[1].split('.')[0].strip()
        
        train_processed['Title'] = train_processed['Name'].apply(extract_title)
        test_processed['Title'] = test_processed['Name'].apply(extract_title)
        
        # Group rare titles
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        
        train_processed['Title'] = train_processed['Title'].map(title_mapping)
        test_processed['Title'] = test_processed['Title'].map(title_mapping)
        
        print("✓ Extracted titles from names")
        
        # 2. Create family size feature
        train_processed['FamilySize'] = train_processed['SibSp'] + train_processed['Parch'] + 1
        test_processed['FamilySize'] = test_processed['SibSp'] + test_processed['Parch'] + 1
        
        # 3. Create IsAlone feature
        train_processed['IsAlone'] = (train_processed['FamilySize'] == 1).astype(int)
        test_processed['IsAlone'] = (test_processed['FamilySize'] == 1).astype(int)
        
        print("✓ Created family size and IsAlone features")
        
        # Handle missing values first (before creating groups)
        print("\nHandling missing values:")
        
        # Fill missing Age with median by Title and Pclass
        for title in train_processed['Title'].unique():
            for pclass in train_processed['Pclass'].unique():
                median_age = train_processed[(train_processed['Title'] == title) & 
                                           (train_processed['Pclass'] == pclass)]['Age'].median()
                
                if not pd.isna(median_age):
                    train_processed.loc[(train_processed['Title'] == title) & 
                                      (train_processed['Pclass'] == pclass) & 
                                      (train_processed['Age'].isna()), 'Age'] = median_age
                    test_processed.loc[(test_processed['Title'] == title) & 
                                     (test_processed['Pclass'] == pclass) & 
                                     (test_processed['Age'].isna()), 'Age'] = median_age
        
        # Fill remaining missing ages with overall median
        train_processed['Age'].fillna(train_processed['Age'].median(), inplace=True)
        test_processed['Age'].fillna(train_processed['Age'].median(), inplace=True)
        
        # Fill missing Embarked with mode
        train_processed['Embarked'].fillna(train_processed['Embarked'].mode()[0], inplace=True)
        test_processed['Embarked'].fillna(train_processed['Embarked'].mode()[0], inplace=True)
        
        # Fill missing Fare with median by Pclass
        for pclass in [1, 2, 3]:
            median_fare = train_processed[train_processed['Pclass'] == pclass]['Fare'].median()
            test_processed.loc[(test_processed['Pclass'] == pclass) & 
                             (test_processed['Fare'].isna()), 'Fare'] = median_fare
        
        print("✓ Filled missing values")
        
        # 4. Create age groups (after filling missing values)
        def categorize_age(age):
            if age < 16:
                return 'Child'
            elif age < 32:
                return 'Young Adult'
            elif age < 48:
                return 'Adult'
            elif age < 64:
                return 'Senior'
            else:
                return 'Elderly'
        
        train_processed['AgeGroup'] = train_processed['Age'].apply(categorize_age)
        test_processed['AgeGroup'] = test_processed['Age'].apply(categorize_age)
        
        print("✓ Created age groups")
        
        # 5. Create fare groups (after filling missing values)
        def categorize_fare(fare):
            if fare < 7.91:
                return 'Low'
            elif fare < 14.45:
                return 'Medium'
            elif fare < 31:
                return 'High'
            else:
                return 'Very High'
        
        train_processed['FareGroup'] = train_processed['Fare'].apply(categorize_fare)
        test_processed['FareGroup'] = test_processed['Fare'].apply(categorize_fare)
        
        print("✓ Created fare groups")
        
        # 6. Create more advanced features
        # Age * Class interaction
        train_processed['Age_Class'] = train_processed['Age'] * train_processed['Pclass']
        test_processed['Age_Class'] = test_processed['Age'] * test_processed['Pclass']
        
        # Fare per person
        train_processed['Fare_Per_Person'] = train_processed['Fare'] / train_processed['FamilySize']
        test_processed['Fare_Per_Person'] = test_processed['Fare'] / test_processed['FamilySize']
        
        # Title and Class interaction
        train_processed['Title_Class'] = train_processed['Title'].astype(str) + '_' + train_processed['Pclass'].astype(str)
        test_processed['Title_Class'] = test_processed['Title'].astype(str) + '_' + test_processed['Pclass'].astype(str)
        
        # Family survival groups
        def family_survival_group(family_size):
            if family_size == 1:
                return 'Alone'
            elif family_size <= 4:
                return 'Small_Family'
            else:
                return 'Large_Family'
        
        train_processed['Family_Survival_Group'] = train_processed['FamilySize'].apply(family_survival_group)
        test_processed['Family_Survival_Group'] = test_processed['FamilySize'].apply(family_survival_group)
        
        # Deck from cabin (first letter)
        train_processed['Deck'] = train_processed['Cabin'].fillna('Unknown').str[0]
        test_processed['Deck'] = test_processed['Cabin'].fillna('Unknown').str[0]
        
        # Group rare decks
        deck_mapping = {'A': 'ABC', 'B': 'ABC', 'C': 'ABC', 'D': 'DE', 'E': 'DE', 
                       'F': 'FG', 'G': 'FG', 'T': 'Other', 'Unknown': 'Unknown'}
        train_processed['Deck'] = train_processed['Deck'].map(deck_mapping)
        test_processed['Deck'] = test_processed['Deck'].map(deck_mapping)
        
        # 7. Advanced feature: Ticket frequency (highly predictive)
        # People with same ticket often traveled together (families/groups)
        ticket_counts = train_processed['Ticket'].value_counts().to_dict()
        train_processed['Ticket_Freq'] = train_processed['Ticket'].map(ticket_counts)
        
        # For test set, use training ticket counts, default to 1 for new tickets
        test_processed['Ticket_Freq'] = test_processed['Ticket'].map(ticket_counts).fillna(1)
        
        # 8. Age and Fare quantile-based binning (more intelligent than fixed ranges)
        train_processed['Age_Quantile'] = pd.qcut(train_processed['Age'], q=5, labels=False, duplicates='drop')
        test_processed['Age_Quantile'] = pd.cut(test_processed['Age'], 
                                               bins=pd.qcut(train_processed['Age'], q=5, retbins=True, duplicates='drop')[1], 
                                               labels=False, include_lowest=True).fillna(2)
        
        train_processed['Fare_Quantile'] = pd.qcut(train_processed['Fare'], q=4, labels=False, duplicates='drop')
        test_processed['Fare_Quantile'] = pd.cut(test_processed['Fare'], 
                                                bins=pd.qcut(train_processed['Fare'], q=4, retbins=True, duplicates='drop')[1], 
                                                labels=False, include_lowest=True).fillna(1)
        
        print("✓ Created advanced interaction features")
        print("✓ Added Ticket_Freq feature (group travel indicator)")
        print("✓ Created quantile-based age and fare binning")
        
        # Feature Selection: Enhanced with new high-impact features
        # Added Ticket_Freq and quantile features for better performance
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 
                   'Title', 'FamilySize', 'Age_Class', 'Fare_Per_Person',
                   'Ticket_Freq', 'Age_Quantile', 'Fare_Quantile']
        
        print("📊 Enhanced Feature Selection Applied:")
        print("   ✅ Kept: High-importance features (>2% importance)")
        print("   ✅ Added: Ticket_Freq, Age_Quantile, Fare_Quantile")
        print("   ❌ Removed: Embarked, Deck, FareGroup, AgeGroup, IsAlone, Family_Survival_Group")
        print("   🎯 Goal: Maximize predictive power while controlling overfitting")
        
        X_train = train_processed[features].copy()
        X_test = test_processed[features].copy()
        y_train = train_processed['Survived']
        
        # Encode categorical variables (simplified set)
        label_encoders = {}
        categorical_features = ['Sex', 'Title']
        
        for feature in categorical_features:
            le = LabelEncoder()
            X_train[feature] = le.fit_transform(X_train[feature])
            X_test[feature] = le.transform(X_test[feature])
            label_encoders[feature] = le
        
        print("✓ Encoded categorical variables")
        
        # Feature scaling for numerical features (enhanced set)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        numerical_features = ['Age', 'Fare', 'Age_Class', 'Fare_Per_Person', 'Ticket_Freq']
        
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])
        
        print("✓ Applied feature scaling to numerical features")
        
        # Split training data for validation with stratification
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        self.X_test = X_test
        self.feature_names = features
        self.label_encoders = label_encoders
        self.scaler = scaler
        
        print(f"\nFinal feature set: {features}")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
    def train_model(self):
        """Train the decision tree model with aggressive hyperparameter tuning."""
        print("\n" + "="*50)
        print("MODEL TRAINING - AGGRESSIVE OPTIMIZATION")
        print("="*50)
        
        # Enhanced parameter grid for maximum performance
        param_grid = {
            'max_depth': [5, 6, 7, 8, 9, 10, 11],  # Expanded range
            'min_samples_split': [4, 6, 8, 10, 12, 15, 20],  # Lower values for fine-tuning
            'min_samples_leaf': [1, 2, 3, 4, 5],  # Include 1 for maximum flexibility
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', 0.7, 0.8, 0.9, None],  # Add percentage options
            'class_weight': [None, 'balanced', {0: 0.6, 1: 1.4}, {0: 0.65, 1: 1.35}, {0: 0.7, 1: 1.3}]  # Custom weights
        }
        
        print("Performing aggressive hyperparameter optimization...")
        print("🎯 Focus: Maximum accuracy with enhanced feature set")
        print(f"🔍 Search space: {len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['criterion']) * len(param_grid['max_features']) * len(param_grid['class_weight'])} combinations")
        
        # Create base decision tree
        dt = DecisionTreeClassifier(random_state=42)
        
        # Use StratifiedKFold for better validation
        cv_strategy = StratifiedKFold(n_splits=12, shuffle=True, random_state=42)
        
        # Perform grid search with enhanced cross-validation
        grid_search = GridSearchCV(
            dt, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(self.X_train, self.y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train the final model
        self.model.fit(self.X_train, self.y_train)
        
        print("✅ Aggressive model optimization completed")
        
    def evaluate_model(self):
        """Evaluate the trained model."""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions with Decision Tree
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        val_accuracy = accuracy_score(self.y_val, y_val_pred)
        
        print("\nDecision Tree Performance (Optimized Feature Set):")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Calculate overfitting metric
        overfitting = train_accuracy - val_accuracy
        print(f"Overfitting Gap: {overfitting:.4f}")
        
        if overfitting < 0.10:
            print("✅ Good generalization (overfitting < 10%)")
        elif overfitting < 0.15:
            print("⚠️  Moderate overfitting (10-15%)")
        else:
            print("❌ High overfitting (>15%)")
        
        # Enhanced cross-validation scores
        cv_strategy = StratifiedKFold(n_splits=12, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv_strategy)
        print(f"\nEnhanced Decision Tree CV scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"CV Score Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")
        
        # Classification report for Decision Tree
        print(f"\nClassification Report (Decision Tree - Validation Set):")
        print(classification_report(self.y_val, y_val_pred))
        
        # Confusion matrix for Decision Tree
        cm = confusion_matrix(self.y_val, y_val_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Create evaluation plots
        self._create_evaluation_plots(cm, feature_importance)
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance
        }
        
    def _create_evaluation_plots(self, confusion_matrix, feature_importance):
        """Create evaluation plots."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Confusion Matrix
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'],
                   ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # 2. Feature Importance
        top_features = feature_importance.head(10)
        axes[1].barh(range(len(top_features)), top_features['importance'])
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features['feature'])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Top 10 Feature Importance')
        axes[1].invert_yaxis()
        
        # 3. Decision Tree Visualization (simplified)
        plot_tree(self.model, feature_names=self.feature_names, 
                 class_names=['Died', 'Survived'], filled=True, rounded=True,
                 max_depth=3, ax=axes[2])
        axes[2].set_title('Decision Tree (Depth=3)')
        
        plt.tight_layout()
        plt.savefig('../results/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_tree(self):
        """Create a detailed visualization of the decision tree."""
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, feature_names=self.feature_names, 
                 class_names=['Died', 'Survived'], filled=True, rounded=True,
                 fontsize=10)
        plt.title('Complete Decision Tree Visualization', fontsize=16, fontweight='bold')
        plt.savefig('../results/decision_tree_full.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_predictions(self):
        """Generate predictions for the test set."""
        print("\n" + "="*50)
        print("GENERATING PREDICTIONS")
        print("="*50)
        
        # Make predictions on test set with Decision Tree
        test_predictions = self.model.predict(self.X_test)
        
        print("Using optimized Decision Tree for final predictions")
        
        # Create submission file
        submission = pd.DataFrame({
            'PassengerId': self.test_data['PassengerId'],
            'Survived': test_predictions
        })
        
        submission.to_csv('../results/titanic_predictions.csv', index=False)
        print("✓ Predictions saved to ../results/titanic_predictions.csv")
        
        # Print prediction summary
        survival_rate = test_predictions.mean()
        print(f"Predicted survival rate on test set: {survival_rate:.2%}")
        
        return submission
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("="*60)
        print("TITANIC SURVIVAL PREDICTION - DECISION TREE ANALYSIS")
        print("="*60)
        
        # Load data
        if not self.load_data():
            return False
            
        # Exploratory data analysis
        self.exploratory_data_analysis()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        results = self.evaluate_model()
        
        # Visualize tree
        self.visualize_tree()
        
        # Generate predictions
        predictions = self.generate_predictions()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Optimized Decision Tree Algorithm Results:")
        print(f"Final Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Cross-Validation Score: {results['cv_scores'].mean():.4f}")
        print(f"Feature Set: {len(self.feature_names)} carefully selected features")
        print(f"Overfitting Control: {(results['train_accuracy'] - results['val_accuracy']):.4f} gap")
        print(f"Check the 'results' folder for generated plots and predictions.")
        
        return True


def main():
    """Main function to run the Titanic analysis."""
    # Create an instance of the TitanicDecisionTree class
    titanic_analyzer = TitanicDecisionTree()
    
    # Run the complete analysis
    success = titanic_analyzer.run_complete_analysis()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
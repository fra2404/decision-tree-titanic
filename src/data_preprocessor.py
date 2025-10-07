#!/usr/bin/env python3
"""
Advanced Data Preprocessing for Titanic ID3 Decision Tree
BUAA Artificial Intelligence Assignment 2

This module implements intelligent data preprocessing while maintaining
the categorical approach required for Prof. Bai Xiao's pure ID3 algorithm.

Key Features:
- Information Gain-based optimal binning
- Advanced outlier detection and capping
- Smart missing value imputation
- Feature engineering with title extraction
- Scientific approach to data optimization

Author: Francesco Albano - lj2506219
Date: October 2025
"""

import pandas as pd
import numpy as np
import math
from scipy import stats

class TitanicDataPreprocessor:
    """
    Advanced data preprocessing for Titanic dataset optimized for ID3 performance.
    
    This class implements intelligent data preprocessing techniques that improve
    ID3 decision tree performance while maintaining algorithm purity.
    """
    
    def __init__(self):
        """Initialize the preprocessor with default parameters."""
        self.age_cuts = None
        self.fare_cuts = None
        self.age_bins = None
        self.fare_bins = None
        self.outlier_threshold = 3.0
        self.max_bins = 6
    
    def print_subheader(self, title):
        """Print a formatted subheader for better output organization."""
        print(f"\n🎯 {title}")
        print("-" * 60)
    
    def detect_and_handle_outliers(self, data, column, method='iqr'):
        """
        Advanced outlier detection and intelligent handling.
        
        Args:
            data (pd.DataFrame): Input dataset
            column (str): Column name to process
            method (str): Detection method ('iqr' or 'percentile')
            
        Returns:
            pd.Series: Cleaned column data with outliers capped
        """
        print(f"   🔍 Outlier analysis for {column}:")
        
        # Initialize cleaned data to handle all cases
        data_cleaned = data[column].copy()
        
        if method == 'iqr':
            # Interquartile Range method
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            print(f"      IQR method: {len(outliers)} outliers detected")
            print(f"      Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Cap outliers instead of removing them (preserves data)
            data_cleaned = data[column].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'percentile':
            # Percentile-based method (more conservative)
            lower_percentile = data[column].quantile(0.05)
            upper_percentile = data[column].quantile(0.95)
            outliers = data[(data[column] < lower_percentile) | (data[column] > upper_percentile)]
            print(f"      Percentile method: {len(outliers)} outliers detected")
            print(f"      Valid range: [{lower_percentile:.2f}, {upper_percentile:.2f}]")
            
            data_cleaned = data[column].clip(lower=lower_percentile, upper=upper_percentile)
        else:
            print(f"      Warning: Unknown method '{method}', returning original data")
        
        print(f"      ✅ Outliers handled with {method}")
        return data_cleaned
    
    def optimal_binning_analysis(self, data, column, target, max_bins=None):
        """
        Find optimal number of bins to maximize Information Gain.
        
        This is the key innovation: instead of arbitrary binning, we test
        all possible binning strategies and choose the one that maximizes
        the Information Gain for the target variable.
        
        Args:
            data (pd.DataFrame): Dataset with column and target
            column (str): Column to optimize binning for
            target (str): Target variable name
            max_bins (int): Maximum number of bins to test
            
        Returns:
            tuple: (best_bins, best_cuts, best_gain)
        """
        if max_bins is None:
            max_bins = self.max_bins
            
        print(f"   📊 Optimal binning analysis for {column}:")
        
        best_bins = 3
        best_gain = -1
        best_cuts = None
        
        for n_bins in range(2, max_bins + 1):
            try:
                # Create bins using quantile-based cutting for balanced distribution
                cuts = pd.qcut(data[column], q=n_bins, duplicates='drop', retbins=True)[1]
                
                # Create temporary categorical variable
                temp_categorical = pd.cut(
                    data[column], 
                    bins=cuts, 
                    labels=[f'Bin_{i+1}' for i in range(len(cuts)-1)], 
                    include_lowest=True
                )
                
                # Calculate Information Gain for this binning strategy
                gain = self._calculate_information_gain_silent(data, temp_categorical, target)
                
                print(f"      {n_bins} bins: IG = {gain:.4f}")
                
                # Track the best performing binning strategy
                if gain > best_gain:
                    best_gain = gain
                    best_bins = n_bins
                    best_cuts = cuts
                    
            except Exception as e:
                print(f"      {n_bins} bins: Error - {str(e)}")
                continue
        
        print(f"      🎯 OPTIMAL: {best_bins} bins (IG = {best_gain:.4f})")
        return best_bins, best_cuts, best_gain
    
    def _calculate_information_gain_silent(self, data, categorical_series, target_col):
        """
        Calculate Information Gain for binning optimization.
        
        Uses the same formula as Prof. Bai Xiao's ID3 algorithm:
        IG(S,A) = E(S) - Σ(|Sv|/|S|) * E(Sv)
        
        Args:
            data (pd.DataFrame): Dataset
            categorical_series (pd.Series): Categorical attribute
            target_col (str): Target variable name
            
        Returns:
            float: Information Gain value
        """
        # Calculate entropy of the whole dataset
        value_counts = data[target_col].value_counts()
        total = len(data)
        total_entropy = 0
        
        for count in value_counts.values:
            probability = count / total
            if probability > 0:
                total_entropy += -probability * math.log2(probability)
        
        # Calculate weighted entropy of subsets
        attribute_values = categorical_series.unique()
        weighted_entropy = 0
        
        for value in attribute_values:
            if pd.isna(value):
                continue
                
            subset_indices = categorical_series == value
            subset = data[subset_indices]
            subset_size = len(subset)
            
            if subset_size == 0:
                continue
                
            # Calculate entropy of this subset
            subset_counts = subset[target_col].value_counts()
            subset_entropy = 0
            
            for count in subset_counts.values:
                probability = count / subset_size
                if probability > 0:
                    subset_entropy += -probability * math.log2(probability)
            
            # Add weighted contribution
            weight = subset_size / total
            weighted_entropy += weight * subset_entropy
        
        return total_entropy - weighted_entropy
    
    def load_and_clean_data(self, data_path="data/"):
        """
        Load and clean the raw Titanic dataset with advanced preprocessing.
        
        Implements intelligent missing value imputation and outlier handling
        while preserving all data points for maximum information retention.
        
        Args:
            data_path (str): Path to data directory
            
        Returns:
            pd.DataFrame: Cleaned dataset ready for categorical conversion
        """
        print("📊 Loading Titanic dataset...")
        
        try:
            train_data = pd.read_csv(f"{data_path}train.csv")
            print(f"✅ Training set loaded: {train_data.shape[0]} examples, {train_data.shape[1]} attributes")
            print("ℹ️  Test set ignored as requested")
            
        except Exception as e:
            print(f"❌ Loading error: {e}")
            return None
        
        self.print_subheader("ADVANCED DATA CLEANING (MAINTAINS PURE ID3)")
        
        # 1. Analyze missing values for transparency
        print("🔍 Missing values analysis:")
        missing_data = train_data.isnull().sum()
        for col, missing in missing_data.items():
            if missing > 0:
                percentage = (missing / len(train_data)) * 100
                print(f"   {col}: {missing} ({percentage:.1f}%)")
        
        # 2. Smart Age imputation using correlated features
        print("\n🎯 Advanced Age handling:")
        age_median_by_group = train_data.groupby(['Pclass', 'Sex'])['Age'].median()
        
        def impute_age_intelligently(row):
            """Impute age based on passenger class and sex correlation."""
            if pd.isna(row['Age']):
                return age_median_by_group.get(
                    (row['Pclass'], row['Sex']), 
                    train_data['Age'].median()
                )
            return row['Age']
        
        train_data['Age'] = train_data.apply(impute_age_intelligently, axis=1)
        train_data['Age'] = self.detect_and_handle_outliers(train_data, 'Age', 'iqr')
        
        # 3. Smart Fare imputation using passenger class correlation
        print("\n🎯 Advanced Fare handling:")
        fare_median_by_class = train_data.groupby('Pclass')['Fare'].median()
        
        def impute_fare_intelligently(row):
            """Impute fare based on passenger class correlation."""
            if pd.isna(row['Fare']):
                return fare_median_by_class.get(row['Pclass'], train_data['Fare'].median())
            return row['Fare']
        
        train_data['Fare'] = train_data.apply(impute_fare_intelligently, axis=1)
        train_data['Fare'] = self.detect_and_handle_outliers(train_data, 'Fare', 'percentile')
        
        # 4. Handle Embarked with mode imputation
        mode_embarked = train_data['Embarked'].mode()[0]
        train_data['Embarked'] = train_data['Embarked'].fillna(mode_embarked)
        
        return train_data
    
    def convert_to_categorical_optimized(self, data):
        """
        Convert numerical data to optimized categorical format for ID3.
        
        This method implements the core innovation: Information Gain-based
        binning optimization and advanced feature engineering.
        
        Args:
            data (pd.DataFrame): Cleaned numerical dataset
            
        Returns:
            tuple: (categorical_data, feature_names)
        """
        self.print_subheader("OPTIMIZED CATEGORICAL DATA CONVERSION")
        
        # Initialize categorical dataset
        categorical_data = pd.DataFrame()
        categorical_data['PassengerId'] = data['PassengerId']
        categorical_data['Survived'] = data['Survived'].map({0: 'No', 1: 'Yes'})
        
        # 1. Sex - already categorical
        categorical_data['Sex'] = data['Sex']
        
        # 2. Passenger Class with descriptive labels
        pclass_mapping = {1: 'First', 2: 'Second', 3: 'Third'}
        categorical_data['Pclass'] = data['Pclass'].map(pclass_mapping)
        
        # 3. OPTIMIZED Age categorization using Information Gain
        target_series = data['Survived'].map({0: 'No', 1: 'Yes'})
        self.age_bins, self.age_cuts, age_gain = self.optimal_binning_analysis(
            pd.DataFrame({'Age': data['Age'], 'Survived': target_series}), 
            'Age', 'Survived'
        )
        
        print(f"   ✅ Age: Optimized binning with {self.age_bins} categories")
        
        # Apply optimized age binning with meaningful labels
        age_labels = ['Child', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior', 'Elderly']
        used_labels = age_labels[:len(self.age_cuts) - 1]
        categorical_data['Age_Group'] = pd.cut(
            data['Age'], 
            bins=self.age_cuts, 
            labels=used_labels, 
            include_lowest=True
        )
        
        # 4. OPTIMIZED Fare categorization using Information Gain
        self.fare_bins, self.fare_cuts, fare_gain = self.optimal_binning_analysis(
            pd.DataFrame({'Fare': data['Fare'], 'Survived': target_series}), 
            'Fare', 'Survived'
        )
        
        print(f"   ✅ Fare: Optimized binning with {self.fare_bins} categories")
        
        # Apply optimized fare binning with descriptive labels
        fare_labels = ['Low', 'Medium', 'High', 'Very_High', 'Luxury', 'Super_Luxury']
        used_labels = fare_labels[:len(self.fare_cuts) - 1]
        categorical_data['Fare_Group'] = pd.cut(
            data['Fare'], 
            bins=self.fare_cuts, 
            labels=used_labels, 
            include_lowest=True
        )
        
        # 5. Enhanced Family Size categorization
        family_size = data['SibSp'] + data['Parch'] + 1
        
        def categorize_family_size(size):
            """Categorize family size with meaningful groups."""
            if size == 1:
                return 'Alone'
            elif size == 2:
                return 'Couple'
            elif size <= 4:
                return 'Small_Family'
            elif size <= 6:
                return 'Medium_Family'
            else:
                return 'Large_Family'
        
        categorical_data['Family_Size_Group'] = family_size.apply(categorize_family_size)
        
        # 6. Embarked with full descriptive names
        embarked_mapping = {
            'C': 'Cherbourg', 
            'Q': 'Queenstown', 
            'S': 'Southampton'
        }
        categorical_data['Embarked'] = data['Embarked'].map(embarked_mapping)
        
        # 7. FEATURE ENGINEERING: Title extraction from Name
        def extract_title_from_name(name):
            """
            Extract social title from passenger name.
            
            This feature captures social class and gender information
            that significantly correlates with survival probability.
            """
            if pd.isna(name):
                return 'Unknown'
            
            # Standard titles
            if 'Mr.' in name:
                return 'Mr'
            elif 'Mrs.' in name or 'Mme.' in name:
                return 'Mrs'
            elif 'Miss.' in name or 'Mlle.' in name:
                return 'Miss'
            elif 'Master.' in name:
                return 'Master'
            # Professional titles
            elif any(title in name for title in ['Dr.', 'Rev.', 'Col.', 'Major.', 'Capt.']):
                return 'Professional'
            # Noble titles
            elif any(title in name for title in ['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.']):
                return 'Noble'
            else:
                return 'Other'
        
        categorical_data['Title_Group'] = data['Name'].apply(extract_title_from_name)
        print("   ✅ Title_Group: New feature extracted from Name")
        
        # 8. Final cleanup - handle any remaining missing values
        for col in categorical_data.columns:
            if categorical_data[col].dtype == 'object':
                categorical_data[col] = categorical_data[col].fillna('Unknown')
        
        # Define final feature set for ID3 algorithm
        feature_names = [
            'Sex', 'Pclass', 'Age_Group', 'Fare_Group', 
            'Family_Size_Group', 'Embarked', 'Title_Group'
        ]
        
        # Summary output
        print(f"\n📊 Optimized categorical dataset:")
        print(f"   Training: {len(categorical_data)} examples")
        print(f"   Categorical features: {feature_names}")
        print(f"\n🎯 IMPORTANT: Only DATA is optimized")
        print(f"   ID3 ALGORITHM remains identical to Prof. Bai Xiao!")
        
        return categorical_data, feature_names
    
    def get_preprocessing_stats(self):
        """
        Get comprehensive statistics about the preprocessing operations.
        
        Returns:
            dict: Dictionary containing preprocessing statistics
        """
        stats = {
            'age_bins': self.age_bins,
            'fare_bins': self.fare_bins,
            'age_cuts': self.age_cuts.tolist() if self.age_cuts is not None else None,
            'fare_cuts': self.fare_cuts.tolist() if self.fare_cuts is not None else None,
            'max_bins_tested': self.max_bins,
            'outlier_threshold': self.outlier_threshold
        }
        return stats
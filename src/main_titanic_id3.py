#!/usr/bin/env python3
"""
Titanic ID3 Pure Optimized - Main Application
BUAA Artificial Intelligence Assignment 2

This is the main application that coordinates the Pure ID3 algorithm
with optimized data preprocessing and comprehensive visualizations.

MODULAR STRUCTURE:
- id3_core_algorithm.py: Pure ID3 algorithm Prof. Bai Xiao
- data_preprocessor.py: Advanced data preprocessing
- visualization_export.py: Visualizations and export
- main_titanic_id3.py: Main coordinator (this file)

Author: Francesco Albano - lj2506219
Date: October 2025
"""

import pandas as pd
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from id3_core_algorithm import PureID3Algorithm
from data_preprocessor import TitanicDataPreprocessor
from visualization_export import TitanicID3Visualizer

class TitanicID3PureOptimizedApplication:
    """
    Main application coordinating Pure ID3 algorithm with optimized data.
    """
    
    def __init__(self, data_path="data/"):
        """Initialize the application."""
        self.data_path = data_path
        
        # Initialize components
        self.preprocessor = TitanicDataPreprocessor()
        self.id3_algorithm = PureID3Algorithm(max_depth=5)  # Optimal depth for visualization
        self.visualizer = TitanicID3Visualizer()
        
        # Data and results
        self.raw_data = None
        self.processed_data = None
        self.feature_names = None
        self.tree = None
        self.predictions_df = None
        self.accuracy = None
        
    def print_header(self, title):
        """Print a formatted header."""
        print("\n" + "="*80)
        print(title.center(80))
        print("="*80)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        self.print_header("DATA LOADING AND PREPROCESSING")
        
        # Load raw data
        self.raw_data = self.preprocessor.load_and_clean_data(self.data_path)
        if self.raw_data is None:
            return False
        
        # Convert to categorical with optimizations
        self.processed_data, self.feature_names = self.preprocessor.convert_to_categorical_optimized(self.raw_data)
        
        print("\n✅ PREPROCESSING COMPLETED")
        print(f"   Final dataset: {len(self.processed_data)} examples")
        print(f"   Features: {len(self.feature_names)} categorical attributes")
        
        return True
    
    def train_id3_model(self):
        """Train the Pure ID3 model."""
        self.print_header("PURE ID3 TRAINING")
        
        # Train the model (silent mode)
        self.tree = self.id3_algorithm.train(
            self.processed_data, 
            self.feature_names, 
            'Survived',
            verbose=False
        )
        
        print("\n✅ ID3 MODEL TRAINED")
        
        return True
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        self.print_header("MODEL EVALUATION")
        
        # Make predictions
        predictions = self.id3_algorithm.predict(self.processed_data, self.feature_names)
        
        # Calculate accuracy
        actual = self.processed_data['Survived'].tolist()
        correct = sum(1 for pred, act in zip(predictions, actual) if pred == act)
        self.accuracy = correct / len(actual)
        
        # Create predictions dataframe
        self.predictions_df = pd.DataFrame({
            'PassengerId': self.processed_data['PassengerId'],
            'Actual_Survived': self.processed_data['Survived'],
            'Predicted_Survived': predictions,
            'Correct_Prediction': [pred == actual for pred, actual in zip(predictions, actual)]
        })
        
        print("📊 EVALUATION RESULTS:")
        print(f"   Correct predictions: {correct}/{len(actual)}")
        print(f"   Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        
        # Show prediction distribution
        pred_dist = pd.Series(predictions).value_counts()
        actual_dist = self.processed_data['Survived'].value_counts()
        
        print("\n📈 Prediction distribution:")
        print(f"   Predicted - No: {pred_dist.get('No', 0)}, Yes: {pred_dist.get('Yes', 0)}")
        print(f"   Actual    - No: {actual_dist.get('No', 0)}, Yes: {actual_dist.get('Yes', 0)}")
        
        return True
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        self.print_header("CREATING VISUALIZATIONS")
        
        # Get tree statistics
        tree_stats = self.id3_algorithm.get_tree_stats()
        
        # Create comprehensive analysis plots
        plot_file = self.visualizer.create_comprehensive_visualizations(
            self.processed_data, tree_stats, self.accuracy, show_popup=True
        )
        
        # Create decision tree diagram
        tree_file = self.visualizer.create_decision_tree_diagram(self.tree)
        
        print("✅ VISUALIZATIONS CREATED:")
        print(f"   Complete analysis: {plot_file}")
        print(f"   Tree diagram: {tree_file}")
        
        return plot_file, tree_file
    
    def export_results(self):
        """Export all results to files."""
        self.print_header("EXPORTING RESULTS")
        
        # Get statistics
        tree_stats = self.id3_algorithm.get_tree_stats()
        rules = self.id3_algorithm.extract_rules()
        preprocessing_stats = self.preprocessor.get_preprocessing_stats()
        
        # Export to CSV files
        exported_files = self.visualizer.export_results_to_csv(
            self.predictions_df, rules, tree_stats, self.accuracy, 
            preprocessing_stats, show_popup=True
        )
        
        print("✅ EXPORT COMPLETED:")
        print(f"   Files exported: {len(exported_files)}")
        
        return exported_files
    
    def print_tree_structure(self):
        """Print compact tree summary instead of full structure."""
        self.print_header("DECISION TREE SUMMARY")
        
        # Get tree statistics
        tree_stats = self.id3_algorithm.get_tree_stats()
        rules = self.id3_algorithm.extract_rules()
        
        print("🌳 ID3 TREE STATISTICS:")
        print(f"   Total Rules: {tree_stats['total_rules']}")
        print(f"   Survival Rules: {tree_stats['survival_rules']}")
        print(f"   Death Rules: {tree_stats['death_rules']}")
        print(f"   Maximum Depth: {tree_stats['tree_depth']}")
        print(f"   Total Nodes: {tree_stats['tree_nodes']}")
        
        print("\n📏 FIRST 5 EXAMPLE RULES:")
        for i, rule in enumerate(rules[:5], 1):
            print(f"   {i}. {rule}")
        
        if len(rules) > 5:
            print(f"   ... and {len(rules) - 5} more rules")
            
        print("\n✅ Complete tree visualized graphically in generated PNG")
    
    def show_final_summary(self):
        """Show comprehensive final summary."""
        tree_stats = self.id3_algorithm.get_tree_stats()
        preprocessing_stats = self.preprocessor.get_preprocessing_stats()
        
        final_results = self.visualizer.print_final_results(
            self.accuracy, tree_stats, preprocessing_stats
        )
        
        return final_results
    
    def run_complete_analysis(self):
        """Run the complete ID3 analysis pipeline."""
        self.print_header("TITANIC ID3 PURE OPTIMIZED - COMPLETE ANALYSIS")
        
        print("🚀 STARTING MODULAR APPLICATION")
        print("📋 Components:")
        print("   • id3_core_algorithm.py: Pure ID3 algorithm")
        print("   • data_preprocessor.py: Advanced preprocessing") 
        print("   • visualization_export.py: Visualizations and export")
        print("   • main_titanic_id3.py: Main coordinator")
        
        try:
            # Step 1: Load and preprocess data
            if not self.load_and_preprocess_data():
                print("❌ Data preprocessing error")
                return False
            
            # Step 2: Train ID3 model
            if not self.train_id3_model():
                print("❌ Training error")
                return False
            
            # Step 3: Evaluate model
            if not self.evaluate_model():
                print("❌ Evaluation error")
                return False
            
            # Step 4: Print tree structure
            self.print_tree_structure()
            
            # Step 5: Create visualizations
            plot_file, tree_file = self.create_visualizations()
            
            # Step 6: Export results
            exported_files = self.export_results()
            
            # Step 7: Show final summary
            final_results = self.show_final_summary()
            
            self.print_header("ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
            print("✅ ALL COMPONENTS EXECUTED CORRECTLY")
            print("✅ CHARTS AND TREE DISPLAYED WITH POPUPS")
            print("✅ RESULTS EXPORTED IN MULTIPLE FORMATS")
            print("✅ PURE ID3 ALGORITHM MAINTAINED")
            print("✅ DATA OPTIMIZED FOR MAXIMUM PERFORMANCE")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR DURING ANALYSIS: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the complete application."""
    print("🎓 BUAA Artificial Intelligence Assignment 2")
    print("👨‍💻 Francesco Albano - lj2506219")
    print("📚 Prof. Bai Xiao - ID3 Decision Tree Algorithm")
    
    # Create and run the application
    app = TitanicID3PureOptimizedApplication()
    
    success = app.run_complete_analysis()
    
    if success:
        print("\n🎉 APPLICATION COMPLETED SUCCESSFULLY!")
        print("📊 Check the displayed charts and exported files!")
    else:
        print("\n❌ APPLICATION FAILED!")
        print("🔧 Check error messages for debugging.")


if __name__ == "__main__":
    main()
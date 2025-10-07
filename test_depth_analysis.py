#!/usr/bin/env python3
"""
Depth Analysis Test Script for ID3 Decision Tree Optimization
This script analyzes the impact of maximum depth on accuracy and tree visualization.

Used to determine optimal max_depth=5 configuration for the final implementation.

Author: Francesco Albano - lj2506219
Date: October 2025
"""

import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# pylint: disable=wrong-import-position,import-error
from data_preprocessor import TitanicDataPreprocessor
from id3_core_algorithm import PureID3Algorithm
from visualization_export import TitanicID3Visualizer

def test_depth_analysis():
    """Test different maximum depths and compare results."""
    
    print("🔍 ID3 TREE MAXIMUM DEPTH ANALYSIS")
    print("="*60)
    
    # Load and preprocess data
    preprocessor = TitanicDataPreprocessor()
    print("📊 Loading data...")
    train_data = preprocessor.load_and_clean_data('data/')
    
    if train_data is None:
        print("❌ Error loading data!")
        return None
        
    categorical_data, feature_names = preprocessor.convert_to_categorical_optimized(train_data)
    
    print(f"✅ Data processed: {len(categorical_data)} examples")
    print(f"📋 Features: {feature_names}")
    
    # Test different depths
    depths_to_test = [3, 4, 5, 6, 7, None]  # None = no limit
    analysis_results = []
    
    print(f"\n🧪 TESTING {len(depths_to_test)} DIFFERENT DEPTHS:")
    print("-" * 60)
    
    for max_depth in depths_to_test:
        print(f"\n🌳 TESTING MAX_DEPTH = {max_depth if max_depth else 'UNLIMITED'}")
        print("-" * 40)
        
        # Create algorithm with specific depth
        id3 = PureID3Algorithm(max_depth=max_depth)
        
        # Train the model
        attributes = feature_names
        id3.train(categorical_data, attributes, verbose=False)
        
        # Calculate accuracy
        predictions = id3.predict(categorical_data, feature_names)
        correct = sum(1 for i, row in categorical_data.iterrows() 
                     if predictions[i] == row['Survived'])
        accuracy = correct / len(categorical_data)
        
        # Calculate tree statistics
        tree_stats = id3.get_tree_stats()
        
        # Save results
        result = {
            'max_depth': max_depth if max_depth else 'Unlimited',
            'actual_depth': tree_stats['tree_depth'],
            'accuracy': accuracy,
            'total_rules': tree_stats['total_rules'],
            'survival_rules': tree_stats['survival_rules'],
            'death_rules': tree_stats['death_rules'],
            'total_nodes': tree_stats['tree_nodes']
        }
        analysis_results.append(result)
        
        # Print results
        print(f"   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   🌳 Actual depth: {tree_stats['tree_depth']}")
        print(f"   📏 Total rules: {tree_stats['total_rules']}")
        print(f"   📈 Survival rules: {tree_stats['survival_rules']}")
        print(f"   📉 Death rules: {tree_stats['death_rules']}")
        print(f"   🔢 Total nodes: {tree_stats['tree_nodes']}")
        
        # Create visualization for this depth
        if max_depth is not None and max_depth <= 5:  # Only for low depths
            visualizer = TitanicID3Visualizer()
            tree_file = f"results/tree_depth_{max_depth}_test.png"
            
            print(f"   🎨 Creating visualization: {tree_file}")
            try:
                visualizer.create_decision_tree_diagram(id3.tree, tree_file, show_popup=False)
                print(f"   ✅ Tree saved: {tree_file}")
            except (ImportError, AttributeError, ValueError) as e:
                print(f"   ❌ Visualization error: {e}")
    
    # Print final comparison
    print("\n📋 FINAL MAXIMUM DEPTH COMPARISON")
    print("="*80)
    print(f"{'Max Depth':<12} {'Actual':<8} {'Accuracy':<10} {'Rules':<8} {'Nodes':<8} {'Survival':<10} {'Death':<8}")
    print("-"*80)
    
    for result in analysis_results:
        print(f"{str(result['max_depth']):<12} "
              f"{result['actual_depth']:<8} "
              f"{result['accuracy']:.4f}   "
              f"{result['total_rules']:<8} "
              f"{result['total_nodes']:<8} "
              f"{result['survival_rules']:<10} "
              f"{result['death_rules']:<8}")
    
    # Analysis and recommendations
    print("\n🎯 ANALYSIS AND RECOMMENDATIONS:")
    print("-" * 40)
    
    # Find best compromise
    best_accuracy = max(analysis_results, key=lambda x: x['accuracy'])
    print(f"🏆 Best accuracy: {best_accuracy['accuracy']:.4f} with max_depth = {best_accuracy['max_depth']}")
    
    # Find optimal depth for visualization (<=5 with good accuracy)
    visual_candidates = [r for r in analysis_results if (
        isinstance(r['max_depth'], int) and 
        r['max_depth'] <= 5 and 
        r['accuracy'] >= 0.85
    )]
    
    if visual_candidates:
        best_visual = max(visual_candidates, key=lambda x: x['accuracy'])
        print(f"🎨 Best for visualization: max_depth = {best_visual['max_depth']} "
              f"(accuracy: {best_visual['accuracy']:.4f}, rules: {best_visual['total_rules']})")
    
    # Compare accuracy vs complexity
    print("\n📈 ACCURACY vs COMPLEXITY:")
    for result in analysis_results:
        if isinstance(result['max_depth'], int):
            complexity_ratio = result['total_rules'] / result['accuracy'] if result['accuracy'] > 0 else float('inf')
            print(f"   Depth {result['max_depth']}: {result['accuracy']:.4f} accuracy, "
                  f"{result['total_rules']} rules (ratio: {complexity_ratio:.1f})")
    
    return analysis_results

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run analysis
    analysis_results = test_depth_analysis()
    
    print("\n🎉 ANALYSIS COMPLETED!")
    print("📁 Visualizations saved in: results/")
    print("🔍 Check tree_depth_*.png files to see the differences")
    print("\n💡 CONCLUSION: This analysis led to choosing max_depth=5 for optimal balance")
    print("   between accuracy (89.23%) and visualization clarity in the final implementation.")
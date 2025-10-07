#!/usr/bin/env python3
"""
Titanic ID3 Core Algorithm - Prof. Bai Xiao Style
BUAA Artificial Intelligence Assignment 2

This module contains the PURE ID3 algorithm implementation
following Prof. Bai Xiao's specifications exactly.

Author: Francesco Albano - lj2506219
Date: October 2025
"""

import pandas as pd
import numpy as np
import math
from collections import Counter

class PureID3Algorithm:
    """
    PURE ID3 Decision Tree Algorithm - Prof. Bai Xiao Style
    No modifications to the core algorithm, only data preprocessing optimizations.
    """
    
    def __init__(self, max_depth=None):
        """Initialize the Pure ID3 Algorithm."""
        self.tree = None
        self.rules = []
        self.max_depth = max_depth
        
    def calculate_entropy(self, data, target_col='Survived', verbose=True):
        """
        Calculate entropy manually following Prof. Bai Xiao formula.
        E(S) = -∑ p_i × log₂(p_i)
        """
        if len(data) == 0:
            return 0
        
        # Count occurrences of each class
        value_counts = data[target_col].value_counts()
        total = len(data)
        
        entropy = 0
        if verbose:
            print(f"   📊 Calculating entropy for {total} examples:")
        
        for value, count in value_counts.items():
            probability = count / total
            if probability > 0:  # Avoid log(0)
                log_prob = math.log2(probability)
                contribution = -probability * log_prob
                entropy += contribution
                if verbose:
                    print(f"      {value}: {count}/{total} = {probability:.3f}, -p×log₂(p) = {contribution:.3f}")
        
        if verbose:
            print(f"   🎯 Entropia totale: {entropy:.3f}")
        return entropy
    
    def calculate_information_gain(self, data, attribute, target_col='Survived', verbose=True):
        """
        Calculate Information Gain manually following Prof. Bai Xiao formula.
        IG(S,A) = E(S) - ∑ (|Sv|/|S|) × E(Sv)
        """
        if verbose:
            print(f"\n🔍 Calcolo Information Gain per '{attribute}':")
        
        # Calculate entropy of the whole dataset
        total_entropy = self.calculate_entropy(data, target_col, verbose)
        total_size = len(data)
        
        # Calculate weighted entropy of subsets
        attribute_values = data[attribute].unique()
        weighted_entropy = 0
        
        if verbose:
            print(f"   📋 Suddivisioni per '{attribute}':")
        
        for value in attribute_values:
            subset = data[data[attribute] == value]
            subset_size = len(subset)
            subset_entropy = self.calculate_entropy(subset, target_col, False)  # Silent for subsets
            weight = subset_size / total_size
            contribution = weight * subset_entropy
            weighted_entropy += contribution
            
            if verbose:
                print(f"      {attribute}={value}: {subset_size} esempi, E={subset_entropy:.3f}, peso={weight:.3f}, contributo={contribution:.3f}")
        
        information_gain = total_entropy - weighted_entropy
        if verbose:
            print(f"   🎯 Information Gain({attribute}) = {total_entropy:.3f} - {weighted_entropy:.3f} = {information_gain:.3f}")
        
        return information_gain
    
    def find_best_attribute(self, data, attributes, target_col='Survived', verbose=True):
        """Find the attribute with the highest Information Gain."""
        if verbose:
            print(f"\n🏆 SELEZIONE MIGLIOR ATTRIBUTO:")
            print(f"   Attributi candidati: {attributes}")
        
        best_attribute = None
        best_gain = -1
        gains = {}
        
        for attribute in attributes:
            gain = self.calculate_information_gain(data, attribute, target_col, verbose)
            gains[attribute] = gain
            
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
        
        if verbose:
            print("\n   📊 Information Gains Summary:")
            sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)
            for i, (attr, gain) in enumerate(sorted_gains):
                prefix = "👑" if i == 0 else "  "
                print(f"   {prefix} {attr}: {gain:.3f}")
            
            print(f"\n   🎯 SELECTED: {best_attribute} (IG = {best_gain:.3f})")
        
        return best_attribute, best_gain
    
    def build_tree(self, data, attributes, target_col='Survived', depth=0, parent_info="Root", verbose=True):
        """Build PURE ID3 decision tree recursively - IDENTICAL to Prof. Bai Xiao."""
        if verbose:
            print(f"\n{'  ' * depth}🌳 NODE CONSTRUCTION - Depth {depth}")
            print(f"{'  ' * depth}   Examples: {len(data)}, Attributes: {attributes}")
            print(f"{'  ' * depth}   Parent node: {parent_info}")
        
        # Base cases - EXACTLY like Prof. Bai Xiao
        target_values = data[target_col].unique()
        
        # If all examples have the same class
        if len(target_values) == 1:
            result = target_values[0]
            if verbose:
                print(f"{'  ' * depth}   ✅ LEAF: All examples are '{result}'")
            return result
        
        # If no more attributes
        if not attributes:
            # Return majority class
            majority_class = data[target_col].mode()[0]
            if verbose:
                print(f"{'  ' * depth}   ✅ LEAF: No attributes left, majority = '{majority_class}'")
            return majority_class
        
        # Check maximum depth constraint
        if self.max_depth is not None and depth >= self.max_depth:
            majority_class = data[target_col].mode()[0]
            if verbose:
                print(f"{'  ' * depth}   ✅ LEAF: Max depth {self.max_depth} reached, majority = '{majority_class}'")
            return majority_class
        
        # Find best attribute using PURE ID3
        best_attribute, best_gain = self.find_best_attribute(data, attributes, target_col, verbose)
        
        # Create tree node
        tree = {best_attribute: {}}
        if verbose:
            print(f"{'  ' * depth}   🎯 NODE CREATED: {best_attribute} (IG={best_gain:.4f})")
        
        # Remove best attribute from remaining attributes
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]
        
        # Split data by best attribute values
        attribute_values = data[best_attribute].unique()
        
        for value in attribute_values:
            if verbose:
                print(f"{'  ' * depth}   🔄 Processing {best_attribute} = {value}")
            subset = data[data[best_attribute] == value]
            
            if len(subset) == 0:
                # No examples with this attribute value
                majority_class = data[target_col].mode()[0]
                tree[best_attribute][value] = majority_class
            else:
                # Recursively build subtree
                subtree = self.build_tree(
                    subset, remaining_attributes, target_col, 
                    depth + 1, f"{best_attribute}={value}", verbose
                )
                tree[best_attribute][value] = subtree
        
        return tree
    
    def train(self, data, feature_names, target_col='Survived', verbose=True):
        """Train the PURE ID3 decision tree model."""
        if verbose:
            print("🎯 ALGORITHM: PURE ID3 - Prof. Bai Xiao")
            print("🎯 NO modifications to core ID3 algorithm!")
            print("\n🎯 TRAINING START")
            print(f"   Dataset: {len(data)} examples")
            print(f"   Features: {feature_names}")
            print(f"   Target: {target_col}")
        
        # Build the tree using PURE ID3
        self.tree = self.build_tree(data, feature_names, target_col, verbose=verbose)
        
        if verbose:
            print("\n✅ TRAINING COMPLETED!")
        
        return self.tree
    
    def predict_single(self, example, tree=None):
        """Predict a single example and return the path taken."""
        if tree is None:
            tree = self.tree
        
        path = []
        current_tree = tree
        
        while isinstance(current_tree, dict):
            # Get the attribute at this node
            attribute = list(current_tree.keys())[0]
            attribute_value = example.get(attribute)
            
            if attribute_value is None:
                return "Unknown", path
            
            path.append(f"{attribute}={attribute_value}")
            
            # Follow the branch
            if attribute_value in current_tree[attribute]:
                current_tree = current_tree[attribute][attribute_value]
            else:
                return "Unknown", path
        
        # We've reached a leaf
        return current_tree, path
    
    def predict(self, data, feature_names):
        """Predict multiple examples."""
        predictions = []
        
        for idx, row in data.iterrows():
            example = {}
            for feature in feature_names:
                example[feature] = row[feature]
            
            prediction, _ = self.predict_single(example)
            predictions.append(prediction)
        
        return predictions
    
    def extract_rules(self, tree=None, path=[], rules=None):
        """Extract IF-THEN rules from the decision tree."""
        if tree is None:
            tree = self.tree
        if rules is None:
            rules = []
        
        if isinstance(tree, str):
            # Leaf node - create rule
            conditions = " AND ".join(path)
            rule = f"IF {conditions} THEN Survived = {tree}"
            rules.append(rule)
            return rules
        
        # Internal node
        for attribute, branches in tree.items():
            for value, subtree in branches.items():
                new_path = path + [f"{attribute} = {value}"]
                self.extract_rules(subtree, new_path, rules)
        
        self.rules = rules
        return rules
    
    def print_tree_structure(self, tree=None, depth=0):
        """Print the tree structure in a readable format."""
        if tree is None:
            tree = self.tree
        
        if isinstance(tree, str):
            return f"🍃 {tree}"
        
        result = ""
        for attribute, branches in tree.items():
            if depth == 0:
                result += f"🌳 ROOT: {attribute}\n"
            
            for value, subtree in branches.items():
                indent = "  " * (depth + 1)
                result += f"{indent}├─ {attribute} = {value}\n"
                
                if isinstance(subtree, str):
                    result += f"{indent}  → 🍃 {subtree}\n"
                else:
                    subtree_str = self.print_tree_structure(subtree, depth + 2)
                    # Add proper indentation to subtree
                    for line in subtree_str.split('\n'):
                        if line.strip():
                            result += f"{indent}  {line}\n"
        
        return result
    
    def get_tree_stats(self):
        """Get statistics about the tree."""
        if self.tree is None:
            return {}
        
        rules = self.extract_rules()
        
        stats = {
            'total_rules': len(rules),
            'survival_rules': len([r for r in rules if 'Yes' in r]),
            'death_rules': len([r for r in rules if 'No' in r]),
            'tree_depth': self._calculate_depth(self.tree),
            'tree_nodes': self._count_nodes(self.tree)
        }
        
        return stats
    
    def _calculate_depth(self, tree, current_depth=0):
        """Calculate the maximum depth of the tree."""
        if isinstance(tree, str):
            return current_depth
        
        max_depth = current_depth
        for attribute, branches in tree.items():
            for value, subtree in branches.items():
                depth = self._calculate_depth(subtree, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _count_nodes(self, tree):
        """Count the total number of nodes in the tree."""
        if isinstance(tree, str):
            return 1  # Leaf node
        
        count = 1  # Current node
        for attribute, branches in tree.items():
            for value, subtree in branches.items():
                count += self._count_nodes(subtree)
        
        return count
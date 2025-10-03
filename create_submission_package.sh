#!/bin/bash

# Titanic Decision Tree Assignment - Package Creation Script
# This script creates a compressed archive ready for submission
# Optimized implementation with 75.37% validation accuracy

echo "📦 Creating assignment submission package..."

# Create a temporary directory for packaging
PACKAGE_DIR="Titanic_Decision_Tree_Assignment_Francesco"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="${PACKAGE_DIR}_${TIMESTAMP}.zip"

# Create the package directory
mkdir -p "$PACKAGE_DIR"

# Copy source code files
echo "📁 Copying source code files..."
cp -r src "$PACKAGE_DIR/"

# Copy data files
echo "📊 Copying data files..."
cp -r data "$PACKAGE_DIR/"

# Copy results and visualizations
echo "📈 Copying results and visualizations..."
cp -r results "$PACKAGE_DIR/"

# Copy documentation
echo "📖 Copying documentation..."
cp Lab_Report_Titanic_Decision_Tree.md "$PACKAGE_DIR/"
cp README.md "$PACKAGE_DIR/"
cp requirements.txt "$PACKAGE_DIR/"

# Copy assignment PDF if exists
if [ -f "Assignment-02.pdf" ]; then
    echo "📄 Copying assignment PDF..."
    cp Assignment-02.pdf "$PACKAGE_DIR/"
fi

# Create the zip archive
echo "🗜️ Creating zip archive: $ARCHIVE_NAME"
zip -r "$ARCHIVE_NAME" "$PACKAGE_DIR"

# Clean up temporary directory
rm -rf "$PACKAGE_DIR"

echo "✅ Package created successfully: $ARCHIVE_NAME"
echo ""
echo "📋 Package contents:"
echo "   ├── src/"
echo "   │   └── titanic_decision_tree.py (Optimized implementation with 75.37% accuracy)"
echo "   ├── data/"
echo "   │   ├── train.csv"
echo "   │   ├── test.csv"
echo "   │   └── gender_submission.csv"
echo "   ├── results/"
echo "   │   ├── eda_plots.png"
echo "   │   ├── correlation_matrix.png"
echo "   │   ├── model_evaluation.png"
echo "   │   ├── decision_tree_full.png"
echo "   │   └── titanic_predictions.csv"
echo "   ├── Lab_Report_Titanic_Decision_Tree.md (Comprehensive report)"
echo "   ├── README.md (Project documentation)"
echo "   ├── requirements.txt (Python dependencies)"
echo "   └── Assignment-02.pdf (Assignment requirements)"
echo ""
echo "🎯 Ready for submission!"
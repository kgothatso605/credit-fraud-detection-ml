#!/bin/bash

# Create directory structure
mkdir -p config
mkdir -p data/{raw,processed}
mkdir -p notebooks
mkdir -p src/{data,features,models,deployment,utils}
mkdir -p tests
mkdir -p models
mkdir -p reports/{figures,results}
mkdir -p deployment/{kubernetes,monitoring}
mkdir -p docs

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/deployment/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# Create .gitkeep files
touch data/.gitkeep
touch models/.gitkeep
touch reports/.gitkeep

echo "✓ Directory structure created!"
echo "✓ Run this script then add your code files"

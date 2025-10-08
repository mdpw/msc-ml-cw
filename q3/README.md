# Genetic Algorithm Project

Manufacturing resource allocation optimization using Genetic Algorithm (GA) and Mixed Integer Programming (MIP) with OR-Library benchmark datasets.

## Quick Start

## 1. Setup
python -m venv manufacturing-env </br>
manufacturing-env\Scripts\activate </br>
pip install -r requirements.txt </br>

## 2. Run Complete Pipeline
## Step 1: Download and Prepare Data
python manufacturing_data_loader.py

## Step 2: Analyze Data (EDA & Visualizations)
python app.py

## Step 3: Run Genetic Algorithm
python genetic_algorithm_solver.py

## Step 4: Run MIP Solver (Optimal Solution)
python mip_solver.py

## Step 5: Compare Results
python compare_ga_mip.py

## 3. Done!
Check ga_results/, mip_results/, comparison_results/, and eda_figures/ folders
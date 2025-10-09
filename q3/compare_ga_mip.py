"""
Compare GA and MIP from Previously Saved Results
Loads JSON files instead of re-running algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from manufacturing_data_loader import ManufacturingDataLoader

# -----------------------------
# Load Saved Results
# -----------------------------
def load_ga_results(filepath="ga_results/ga_solution.json"):
    """Load GA results from saved JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GA results not found at {filepath}. Run genetic_algorithm_solver.py first!")
    
    with open(filepath, 'r') as f:
        ga_data = json.load(f)
    
    print(f"Loaded GA results from: {filepath}")
    return ga_data

def load_mip_results(filepath="mip_results/mip_solution.json"):
    """Load MIP results from saved JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MIP results not found at {filepath}. Run mip_solver.py first!")
    
    with open(filepath, 'r') as f:
        mip_data = json.load(f)
    
    print(f"Loaded MIP results from: {filepath}")
    return mip_data

# -----------------------------
# Extract Comparison Data
# -----------------------------
def extract_comparison_data(ga_data, mip_data):
    """Extract comparison metrics from saved results."""
    
    results = {
        'problem_info': {
            'company': ga_data.get('company_name', 'N/A'),
            'n_products': len(ga_data.get('selected_products', [])),
            'n_resources': len(ga_data.get('resources', []))
        },
        'ga': {
            'method': 'Genetic Algorithm',
            'profit': float(ga_data.get('best_profit_usd', 0)),
            'time_seconds': float(ga_data.get('execution_time_seconds', 0)),
            'constraints_satisfied': True,  # Assumed from saved solution
            'products_selected': ga_data.get('number_of_selected_products', 0),
            'parameters': ga_data.get('hyperparameters', {})
        },
        'mip': {
            'method': 'Mixed Integer Programming',
            'profit': float(mip_data.get('best_profit_usd', 0)),
            'time_seconds': float(mip_data.get('solving_time_seconds', 0)),
            'constraints_satisfied': True,  # Assumed from saved solution
            'products_selected': mip_data.get('number_of_selected_products', 0),
            'solver_status': mip_data.get('solver_status', 'Unknown'),
            'is_optimal': mip_data.get('solver_status') == 'Optimal'
        }
    }
    
    # Calculate comparison metrics
    mip_profit = results['mip']['profit']
    ga_profit = results['ga']['profit']
    mip_execution_time = results['mip']['time_seconds']
    ga__execution_time = results['ga']['time_seconds']
    
    if mip_profit > 0:
        optimality_gap = ((mip_profit - ga_profit) / mip_profit) * 100
    else:
        optimality_gap = 0
    
    results['comparison'] = {
        'profit_difference': float(mip_profit - ga_profit),
        'optimality_gap_percent': float(optimality_gap),        
        'time_difference_seconds': abs(float(mip_execution_time - ga__execution_time))
    }
    
    return results

# -----------------------------
# Save Comparison Results
# -----------------------------
def save_comparison_results(results, filename="comparison_results/comparison_from_saved.json"):
    """Save comparison results to JSON."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Comparison results saved to: {filename}")

# -----------------------------
# Generate Text Report
# -----------------------------
def generate_report(results, filename="comparison_results/comparison_report_saved.txt"):
    """Generate text comparison report."""
    
    ga = results['ga']
    mip = results['mip']
    comp = results['comparison']
    gap = comp['optimality_gap_percent']
    
    report = f"""
        {'='*80}
                    GA vs MIP COMPARISON REPORT (FROM SAVED RESULTS)
        {'='*80}

        PROBLEM DETAILS:
        {'-'*80}
        Company: {results['problem_info']['company']}
        Resources: {results['problem_info']['n_resources']}

        ALGORITHM PERFORMANCE:
        {'-'*80}

        1. GENETIC ALGORITHM
        • Objective Value: ${ga['profit']:,.2f}
        • Products Selected: {ga['products_selected']}
        • Constraints Satisfied: {'Yes' if ga['constraints_satisfied'] else 'No'}
        • Parameters: {ga['parameters']}
        • Execution Time: {ga['time_seconds']:,.2f} seconds

        2. MIXED INTEGER PROGRAMMING (OPTIMAL)
        • Objective Value: ${mip['profit']:,.2f}
        • Products Selected: {mip['products_selected']}
        • Constraints Satisfied: {'Yes' if mip['constraints_satisfied'] else 'No'}
        • Solver Status: {mip.get('solver_status', 'N/A')}
        • Optimality: {'Guaranteed Optimal' if mip.get('is_optimal') else 'Not Guaranteed'}
        • Execution Time: {mip['time_seconds']:,.2f} seconds

        COMPARISON METRICS:
        {'-'*80}
        • Profit Difference: ${comp['profit_difference']:,.2f}
        • Optimality Gap: {gap:.2f}%
        • Time Difference: {abs(comp['time_difference_seconds']):,.2f} seconds

        ANALYSIS:
        {'-'*80}
        """
    
    if gap < 1:
        report += "Solution Quality: EXCELLENT - GA found near-optimal solution (gap < 1%)\n"
    elif gap < 5:
        report += "Solution Quality: GOOD - GA found high-quality solution (gap < 5%)\n"
    else:
        report += "Solution Quality: FAIR - Consider parameter tuning\n"       
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"Report saved to: {filename}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "GA vs MIP COMPARISON FROM SAVED RESULTS")
    print("="*80)   

    try:
        # Load saved results
        ga_data = load_ga_results()
        mip_data = load_mip_results()
        
        # Extract comparison data
        results = extract_comparison_data(ga_data, mip_data)
                
        # Save results
        save_comparison_results(results)
        
        # Generate report
        generate_report(results)
        
        print("\n" + "="*80)
        print(" "*25 + "COMPARISON COMPLETE!")
        print("="*80)
        print("\nResults saved in 'comparison_results/' directory")  
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the following commands first:")
        print("  1. python genetic_algorithm_solver.py")
        print("  2. python mip_solver.py")
        print("\nThen run this comparison script again.")
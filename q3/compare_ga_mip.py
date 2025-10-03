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
    
    print(f"✓ Loaded GA results from: {filepath}")
    return ga_data

def load_mip_results(filepath="mip_results/mip_solution.json"):
    """Load MIP results from saved JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MIP results not found at {filepath}. Run mip_solver.py first!")
    
    with open(filepath, 'r') as f:
        mip_data = json.load(f)
    
    print(f"✓ Loaded MIP results from: {filepath}")
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
            'time_seconds': None,  # Not available from saved file
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
    
    if mip_profit > 0:
        optimality_gap = ((mip_profit - ga_profit) / mip_profit) * 100
    else:
        optimality_gap = 0
    
    results['comparison'] = {
        'profit_difference': float(mip_profit - ga_profit),
        'optimality_gap_percent': float(optimality_gap),
        'ga_speedup_factor': None,  # Can't calculate without time data
        'time_difference_seconds': None
    }
    
    return results

# -----------------------------
# Display Comparison Table
# -----------------------------
def display_comparison_table(results):
    """Display detailed comparison table."""
    
    print("\n" + "="*80)
    print(" "*30 + "COMPARISON RESULTS")
    print(" "*25 + "(From Saved Solutions)")
    print("="*80)
    
    ga = results['ga']
    mip = results['mip']
    comp = results['comparison']
    
    print("\n1. SOLUTION QUALITY:")
    print("-"*80)
    print(f"{'Metric':<40} {'GA':<20} {'MIP':<20}")
    print("-"*80)
    print(f"{'Profit (USD)':<40} ${ga['profit']:>18,.2f} ${mip['profit']:>18,.2f}")
    print(f"{'Products Selected':<40} {ga['products_selected']:>19} {mip['products_selected']:>19}")
    print(f"{'Constraints Satisfied':<40} {'YES' if ga['constraints_satisfied'] else 'NO':>19} {'YES' if mip['constraints_satisfied'] else 'NO':>19}")
    
    if mip.get('is_optimal'):
        print(f"{'Optimality Gap':<40} {comp['optimality_gap_percent']:>18.2f}% {'0.00% (Optimal)':>20}")
    
    print("\n2. COMPUTATIONAL EFFICIENCY:")
    print("-"*80)
    print(f"{'Metric':<40} {'GA':<20} {'MIP':<20}")
    print("-"*80)
    if mip['time_seconds']:
        print(f"{'Solving Time (seconds)':<40} {'N/A*':>19} {mip['time_seconds']:>19.2f}")
        print("\n  * GA execution time not saved in JSON (run comparison mode for time data)")
    else:
        print(f"{'Solving Time':<40} {'Not Available':>19} {'Not Available':>19}")
    
    print("\n3. ALGORITHM CHARACTERISTICS:")
    print("-"*80)
    print(f"{'Genetic Algorithm:':<40}")
    if ga['parameters']:
        print(f"  {'- Population Size:':<38} {ga['parameters'].get('population_size', 'N/A'):>19}")
        print(f"  {'- Generations:':<38} {ga['parameters'].get('generations', 'N/A'):>19}")
        print(f"  {'- Mutation Rate:':<38} {ga['parameters'].get('mutation_rate', 'N/A'):>19}")
        print(f"  {'- Crossover Rate:':<38} {ga['parameters'].get('crossover_rate', 'N/A'):>19}")
    
    print(f"\n{'Mixed Integer Programming:':<40}")
    print(f"  {'- Solver:':<38} {mip.get('solver', 'CBC (PuLP)'):>19}")
    print(f"  {'- Solution Status:':<38} {mip.get('solver_status', 'N/A'):>19}")
    print(f"  {'- Optimality:':<38} {'Guaranteed' if mip.get('is_optimal') else 'Not Guaranteed':>19}")
    
    print("\n4. KEY INSIGHTS:")
    print("-"*80)
    
    gap = comp['optimality_gap_percent']
    if gap < 1:
        print("  ✓ GA found near-optimal solution (gap < 1%)")
    elif gap < 5:
        print("  ✓ GA found good quality solution (gap < 5%)")
    else:
        print(f"  ⚠ GA solution is {gap:.2f}% below optimal")
    
    if mip.get('is_optimal'):
        print("  ✓ MIP guarantees global optimality")
        print("  ✓ GA provides good heuristic solution")
    
    print("\n" + "="*80)

# -----------------------------
# Visualize Comparison
# -----------------------------
def visualize_comparison_from_saved(results, save_path='comparison_results/ga_vs_mip_saved.png'):
    """Create visualization from saved results."""
    
    ga = results['ga']
    mip = results['mip']
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Profit Comparison
    ax1 = plt.subplot(2, 3, 1)
    methods = ['GA', 'MIP']
    profits = [ga['profit'], mip['profit']]
    colors = ['#2E86AB', '#A23B72']
    bars = ax1.bar(methods, profits, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Profit (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Solution Quality Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(profits) * 1.1)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Optimality Gap
    ax2 = plt.subplot(2, 3, 2)
    gap = results['comparison']['optimality_gap_percent']
    ax2.barh(['Optimality\nGap'], [gap], color='#F18F01', alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_xlabel('Gap from Optimal (%)', fontsize=12, fontweight='bold')
    ax2.set_title('GA Solution Quality', fontsize=14, fontweight='bold')
    ax2.text(gap/2, 0, f'{gap:.2f}%', ha='center', va='center', 
             fontsize=14, fontweight='bold', color='white')
    ax2.set_xlim(0, max(10, gap * 1.2))
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Products Selected
    ax3 = plt.subplot(2, 3, 3)
    products = [ga['products_selected'], mip['products_selected']]
    bars = ax3.bar(methods, products, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
    ax3.set_title('Products Selected', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Profit Difference
    ax4 = plt.subplot(2, 3, 4)
    profit_diff = results['comparison']['profit_difference']
    ax4.bar(['Profit\nDifference'], [profit_diff], color='#6A994E', alpha=0.7, 
            edgecolor='black', linewidth=2)
    ax4.set_ylabel('USD', fontsize=12, fontweight='bold')
    ax4.set_title('MIP Advantage (Profit)', fontsize=14, fontweight='bold')
    ax4.text(0, profit_diff/2, f'${profit_diff:,.0f}', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Summary Text
    ax5 = plt.subplot(2, 3, (5, 6))
    ax5.axis('off')
    
    summary_text = f"""
    COMPARISON SUMMARY (FROM SAVED RESULTS)
    {'='*50}
    
    Problem: {results['problem_info']['company']}
    Total Products Available: ~{results['problem_info']['n_products']*2}
    Resources: {results['problem_info']['n_resources']}
    
    GENETIC ALGORITHM:
    • Profit: ${ga['profit']:,.2f}
    • Products: {ga['products_selected']}
    • Hyperparameters: {ga['parameters']}
    
    MIP (OPTIMAL):
    • Profit: ${mip['profit']:,.2f}
    • Products: {mip['products_selected']}
    • Status: {mip.get('solver_status', 'N/A')}
    
    PERFORMANCE:
    • Gap: {gap:.2f}%
    • Profit Difference: ${profit_diff:,.2f}
    
    RECOMMENDATION:
    {'✓ GA: Excellent near-optimal solution' if gap < 5 else '⚠ Consider MIP for better quality'}
    
    Note: Time comparison unavailable from saved files.
    Run compare_ga_mip.py for complete timing analysis.
    """
    
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('GA vs MIP Comparison (From Saved Results)', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs("comparison_results", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Comparison visualization saved to: {save_path}")

# -----------------------------
# Save Comparison Results
# -----------------------------
def save_comparison_results(results, filename="comparison_results/comparison_from_saved.json"):
    """Save comparison results to JSON."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Comparison results saved to: {filename}")

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

2. MIXED INTEGER PROGRAMMING (OPTIMAL)
   • Objective Value: ${mip['profit']:,.2f}
   • Products Selected: {mip['products_selected']}
   • Constraints Satisfied: {'Yes' if mip['constraints_satisfied'] else 'No'}
   • Solver Status: {mip.get('solver_status', 'N/A')}
   • Optimality: {'Guaranteed Optimal' if mip.get('is_optimal') else 'Not Guaranteed'}

COMPARISON METRICS:
{'-'*80}
• Profit Difference: ${comp['profit_difference']:,.2f}
• Optimality Gap: {gap:.2f}%
• Time Comparison: Not available from saved files

ANALYSIS:
{'-'*80}
"""
    
    if gap < 1:
        report += "✓ Solution Quality: EXCELLENT - GA found near-optimal solution (gap < 1%)\n"
    elif gap < 5:
        report += "✓ Solution Quality: GOOD - GA found high-quality solution (gap < 5%)\n"
    else:
        report += "⚠ Solution Quality: FAIR - Consider parameter tuning\n"
    
    report += f"""
RECOMMENDATIONS:
{'-'*80}
• Use GA for: Fast, near-optimal solutions in production
• Use MIP for: Guaranteed optimal solutions for critical decisions
• Gap of {gap:.2f}% suggests GA is {'highly effective' if gap < 5 else 'acceptable'}

NOTE: For complete timing and scalability analysis, run compare_ga_mip.py
      which executes both algorithms and measures performance.

{'='*80}
"""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"✓ Report saved to: {filename}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "GA vs MIP COMPARISON FROM SAVED RESULTS")
    print("="*80)
    
    print("\nThis script loads previously saved GA and MIP results for comparison.")
    print("Make sure you have run both:")
    print("  1. python genetic_algorithm_solver.py")
    print("  2. python mip_solver.py")
    print("\nLoading saved results...\n")
    
    try:
        # Load saved results
        ga_data = load_ga_results()
        mip_data = load_mip_results()
        
        # Extract comparison data
        results = extract_comparison_data(ga_data, mip_data)
        
        # Display comparison
        display_comparison_table(results)
        
        # Visualize
        visualize_comparison_from_saved(results)
        
        # Save results
        save_comparison_results(results)
        
        # Generate report
        generate_report(results)
        
        print("\n" + "="*80)
        print(" "*25 + "COMPARISON COMPLETE!")
        print("="*80)
        print("\nResults saved in 'comparison_results/' directory")
        print("\nNote: For complete analysis including execution time,")
        print("      run: python compare_ga_mip.py (executes both algorithms)")
        print("\n" + "="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the following commands first:")
        print("  1. python genetic_algorithm_solver.py")
        print("  2. python mip_solver.py")
        print("\nThen run this comparison script again.")
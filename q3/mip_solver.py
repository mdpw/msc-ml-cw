"""
Mixed Integer Programming (MIP) Solver for Manufacturing Resource Allocation
Can be run standalone or imported by comparison module
"""

import numpy as np
import pulp
import time
import json
import os
from manufacturing_data_loader import ManufacturingDataLoader

# -----------------------------
# MIP Formulation
# -----------------------------
def formulate_mip_problem(instance):
    """
    Formulate the manufacturing problem as a Mixed Integer Programming (MIP) problem.
    
    Mathematical Formulation:
    -------------------------
    Decision Variables:
        x_i ∈ {0, 1} for i = 1, 2, ..., n_products
        where x_i = 1 if product i is selected, 0 otherwise
    
    Objective Function:
        Maximize: Σ(profit_i * x_i) for all i
    
    Constraints:
        For each resource r:
            Σ(resource_consumption_r,i * x_i) ≤ capacity_r
    
    Returns:
        prob: PuLP problem object
        variables: Dictionary of decision variables
    """
    
    n_products = instance['n_products']
    n_resources = instance['n_resources']
    
    # Create the optimization problem (Maximization)
    prob = pulp.LpProblem("Manufacturing_Resource_Allocation", pulp.LpMaximize)
    
    # Decision Variables: x[i] = 1 if product i is selected, 0 otherwise
    x = {}
    for i in range(n_products):
        x[i] = pulp.LpVariable(f"x_{i}", cat='Binary')
    
    # Objective Function: Maximize total profit
    prob += pulp.lpSum([instance['profit_margins_usd'][i] * x[i] 
                        for i in range(n_products)]), "Total_Profit"
    
    # Constraints: Resource capacity constraints
    for r in range(n_resources):
        prob += (
            pulp.lpSum([instance['resource_consumption'][r][i] * x[i] 
                       for i in range(n_products)]) 
            <= instance['resource_capacities'][r],
            f"Resource_{instance['resource_names'][r]}_Capacity"
        )
    
    return prob, x

# -----------------------------
# Solve MIP Problem
# -----------------------------
def solve_mip(instance, solver_name='CBC', time_limit=300, verbose=True):
    """
    Solve the MIP problem using PuLP.
    
    Args:
        instance: Problem instance data
        solver_name: Solver to use ('CBC', 'GLPK', 'GUROBI', 'CPLEX')
        time_limit: Maximum solving time in seconds
        verbose: Print solving progress
    
    Returns:
        solution: Binary array of selected products
        objective_value: Total profit
        solve_time: Time taken to solve
        status: Solver status
    """
    
    if verbose:
        print("\n" + "="*70)
        print(" "*20 + "MIP PROBLEM FORMULATION")
        print("="*70)
        print(f"\nProblem: {instance['company_name']}")
        print(f"Products: {instance['n_products']}")
        print(f"Resources: {instance['n_resources']}")
        print(f"Solver: {solver_name}")
        print(f"Time Limit: {time_limit} seconds")
    
    # Formulate the problem
    prob, x = formulate_mip_problem(instance)
    
    if verbose:
        print("\nMIP Formulation:")
        print("-" * 70)
        print(f"Decision Variables: {instance['n_products']} binary variables")
        print(f"Objective: Maximize Σ(profit_i * x_i)")
        print(f"Constraints: {instance['n_resources']} resource capacity constraints")
        print("-" * 70)
    
    # Select solver
    if solver_name.upper() == 'CBC':
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
    elif solver_name.upper() == 'GLPK':
        solver = pulp.GLPK_CMD(timeLimit=time_limit, msg=verbose)
    else:
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
    
    # Solve the problem
    if verbose:
        print("\nSolving MIP problem...")
        print("-" * 70)
    
    start_time = time.time()
    prob.solve(solver)
    solve_time = time.time() - start_time
    
    # Extract solution
    solution = np.zeros(instance['n_products'], dtype=int)
    for i in range(instance['n_products']):
        if x[i].varValue is not None:
            solution[i] = int(x[i].varValue)
    
    objective_value = pulp.value(prob.objective) if prob.objective else 0
    status = pulp.LpStatus[prob.status]
    
    if verbose:
        print(f"\nSolver Status: {status}")
        print(f"Solving Time: {solve_time:.2f} seconds")
        print(f"Objective Value: ${objective_value:,.2f}")
        print(f"Products Selected: {int(np.sum(solution))}")
        print("="*70)
    
    return solution, objective_value, solve_time, status

# -----------------------------
# Evaluate MIP Solution
# -----------------------------
def evaluate_mip_solution(solution, instance):
    """Evaluate MIP solution and check constraints."""
    profits = np.array(instance['profit_margins_usd'])
    capacities = np.array(instance['resource_capacities'])
    resource_consumption = np.array(instance['resource_consumption'])
    
    total_profit = np.sum(solution * profits)
    constraints_ok = True
    usage_report = []
    
    print("\n" + "="*70)
    print(" "*20 + "MIP SOLUTION EVALUATION")
    print("="*70)
    
    for r in range(len(capacities)):
        usage = np.sum(solution * resource_consumption[r])
        feasible = usage <= capacities[r]
        utilization = (usage / capacities[r]) * 100 if capacities[r] > 0 else 0
        
        status = "✓ SATISFIED" if feasible else "✗ VIOLATED"
        if not feasible:
            constraints_ok = False
        
        usage_report.append({
            'name': instance['resource_names'][r],
            'capacity': capacities[r],
            'used': usage,
            'utilization': utilization,
            'unit': instance['resource_units'][r],
            'satisfied': feasible
        })
        
        print(f"\nResource: {instance['resource_names'][r]}")
        print(f"  Capacity: {capacities[r]:,.2f} {instance['resource_units'][r]}")
        print(f"  Used: {usage:,.2f} {instance['resource_units'][r]}")
        print(f"  Utilization: {utilization:.2f}%")
        print(f"  Status: {status}")
    
    print("\n" + "="*70)
    print(f"OBJECTIVE VALUE (Total Profit): ${total_profit:,.2f}")
    print(f"ALL CONSTRAINTS SATISFIED: {'YES ✓' if constraints_ok else 'NO ✗'}")
    print(f"NUMBER OF SELECTED PRODUCTS: {int(np.sum(solution))}")
    print("="*70 + "\n")
    
    return total_profit, constraints_ok, usage_report

# -----------------------------
# Save MIP Solution
# -----------------------------
def save_mip_solution(instance, solution, profit, solve_time, status, 
                      filename="mip_solution.json"):
    """Save MIP solution to JSON file."""
    selected_products = [
        {
            "sku": sku,
            "category": category,
            "profit_usd": float(profit_val),
            "resource_usage": {
                instance['resource_names'][r]: float(instance['resource_consumption'][r][i])
                for r in range(instance['n_resources'])
            }
        }
        for i, (sku, category, profit_val, bit) in enumerate(
            zip(instance['product_skus'],
                instance['product_categories'],
                instance['profit_margins_usd'],
                solution)
        ) if bit == 1
    ]
    
    result = {
        "problem_number": instance["problem_number"],
        "company_name": instance["company_name"],
        "planning_period": instance["planning_period"],
        "optimization_method": "Mixed Integer Programming (MIP)",
        "solver": "PuLP with CBC",
        "solver_status": status,
        "solving_time_seconds": float(solve_time),
        "best_profit_usd": float(profit),
        "number_of_selected_products": int(np.sum(solution)),
        "selected_products": selected_products,
        "resources": [
            {
                "name": instance['resource_names'][i],
                "capacity": float(instance['resource_capacities'][i]),
                "used": float(np.sum(solution * instance['resource_consumption'][i])),
                "unit": instance['resource_units'][i],
                "utilization_percent": float((np.sum(solution * instance['resource_consumption'][i]) / 
                                             instance['resource_capacities'][i]) * 100)
            }
            for i in range(instance['n_resources'])
        ]
    }
    
    os.makedirs("mip_results", exist_ok=True)
    filepath = os.path.join("mip_results", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ MIP solution saved to: {filepath}")

# -----------------------------
# Display MIP Formulation
# -----------------------------
def display_mip_formulation(instance):
    """Display the mathematical formulation of the MIP problem."""
    print("\n" + "="*70)
    print(" "*15 + "MATHEMATICAL FORMULATION (MIP)")
    print("="*70)
    
    print("\n📊 DECISION VARIABLES:")
    print("-" * 70)
    print(f"  x_i ∈ {{0, 1}} for i = 1, 2, ..., {instance['n_products']}")
    print("  where x_i = 1 if product i is selected for production")
    print("        x_i = 0 otherwise")
    
    print("\n🎯 OBJECTIVE FUNCTION:")
    print("-" * 70)
    print("  Maximize: Z = Σ(profit_i × x_i)")
    print(f"  where profit_i are the profit margins for {instance['n_products']} products")
    
    print("\n⚙️ CONSTRAINTS:")
    print("-" * 70)
    print(f"  Resource Capacity Constraints ({instance['n_resources']} constraints):")
    for r in range(instance['n_resources']):
        print(f"    Σ(resource_consumption_{r},i × x_i) ≤ {instance['resource_capacities'][r]}")
        print(f"      [{instance['resource_names'][r]}: {instance['resource_units'][r]}]")
    
    print("\n  Non-negativity and Binary Constraints:")
    print("    x_i ∈ {0, 1} for all i")
    
    print("\n" + "="*70 + "\n")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "MIXED INTEGER PROGRAMMING SOLVER")
    print(" "*20 + "Manufacturing Resource Allocation")
    print("="*70)
    
    # Load data from .npz file using ManufacturingDataLoader
    print("\nLoading problem instance from saved file...")
    loader = ManufacturingDataLoader()
    
    try:
        instance = loader.load_manufacturing_instance('manufacturing_problem_1.npz')
        print("Successfully loaded from manufacturing_problem_1.npz")
    except FileNotFoundError:
        print("Error: manufacturing_problem_1.npz not found!")
        print("Please run manufacturing_data_loader.py first to create the data file.")
        exit(1)
    
    print("\n" + "-"*70)
    print("PROBLEM INSTANCE DETAILS:")
    print("-"*70)
    print(f"Company: {instance['company_name']}")
    print(f"Planning Period: {instance['planning_period']}")
    print(f"Number of Products: {instance['n_products']}")
    print(f"Number of Resources: {instance['n_resources']}")
    print(f"Resources: {', '.join(instance['resource_names'])}")
    print("-"*70)
    
    # Display mathematical formulation
    display_mip_formulation(instance)
    
    # Solve MIP
    solution, objective_value, solve_time, status = solve_mip(
        instance, 
        solver_name='CBC',
        time_limit=300,
        verbose=True
    )
    
    # Evaluate solution
    total_profit, constraints_ok, usage_report = evaluate_mip_solution(solution, instance)
    
    # Display selected products
    selected_products = [
        (sku, cat) 
        for sku, cat, bit in zip(instance['product_skus'], 
                                  instance['product_categories'], 
                                  solution) 
        if bit == 1
    ]
    
    print("\nSELECTED PRODUCTS:")
    print("-"*70)
    num_to_show = min(15, len(selected_products))
    for i, (sku, cat) in enumerate(selected_products[:num_to_show], 1):
        print(f"  {i:2d}. {sku} ({cat})")
    
    if len(selected_products) > num_to_show:
        print(f"  ... and {len(selected_products) - num_to_show} more products")
    
    print(f"\nTotal Selected: {len(selected_products)} products")
    print("-"*70)
    
    # Save solution
    save_mip_solution(instance, solution, total_profit, solve_time, status)
    
    print("\n" + "="*70)
    print(" "*25 + "MIP OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"\n✓ Optimal Profit: ${total_profit:,.2f}")
    print(f"✓ Solving Time: {solve_time:.2f} seconds")
    print(f"✓ Solver Status: {status}")
    print(f"✓ Constraints Satisfied: {'YES' if constraints_ok else 'NO'}")
    print(f"✓ Products Selected: {int(np.sum(solution))} out of {instance['n_products']}")
    print(f"✓ Results saved in 'mip_results/' directory")
    print("\n" + "="*70 + "\n")
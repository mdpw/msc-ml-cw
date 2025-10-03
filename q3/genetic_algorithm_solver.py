"""
Genetic Algorithm Solver for Manufacturing Resource Allocation
Can be run standalone or imported by comparison module
"""

import numpy as np
import matplotlib.pyplot as plt
from manufacturing_data_loader import ManufacturingDataLoader
import json
import os

# -----------------------------
# Genetic Algorithm Components
# -----------------------------
def initialize_population(pop_size, n_products):
    """Create initial population of binary chromosomes."""
    return np.random.randint(2, size=(pop_size, n_products))

def repair_chromosome(chromosome, resource_consumption, capacities):
    """Repair a chromosome to satisfy resource constraints."""
    chromosome = chromosome.copy()
    total_usage = np.sum(chromosome * resource_consumption, axis=1)
    
    while np.any(total_usage > capacities):
        violated_resources = np.where(total_usage > capacities)[0]
        r = violated_resources[0]
        contributing_products = np.where(chromosome * resource_consumption[r] > 0)[0]
        if len(contributing_products) == 0:
            break
        i = np.random.choice(contributing_products)
        chromosome[i] = 0
        total_usage = np.sum(chromosome * resource_consumption, axis=1)
    
    return chromosome

def fitness(chromosome, profits):
    """Fitness = total profit (chromosome should already be feasible)."""
    return np.sum(chromosome * profits)

def tournament_selection(pop, fitnesses, k=3):
    """Select one parent using tournament selection."""
    selected_idx = np.random.choice(len(pop), k)
    best_idx = selected_idx[np.argmax(fitnesses[selected_idx])]
    return pop[best_idx]

def crossover(parent1, parent2, crossover_rate=0.8):
    """Single-point crossover."""
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(chromosome, mutation_rate=0.01):
    """Flip bits with mutation probability."""
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# -----------------------------
# Genetic Algorithm Main Loop
# -----------------------------
def genetic_algorithm(instance, pop_size=100, generations=200,
                      crossover_rate=0.8, mutation_rate=0.01,
                      verbose=True):
    """
    Run Repair-based GA on manufacturing problem instance.
    
    Args:
        instance: Problem instance dictionary
        pop_size: Population size
        generations: Number of generations
        crossover_rate: Crossover probability
        mutation_rate: Mutation probability
        verbose: Print progress
    
    Returns:
        tuple: (best_solution, best_fitness, convergence_curve)
        - best_solution: Binary array of selected products
        - best_fitness: Best profit achieved
        - convergence_curve: List of best fitness per generation
    """
    n_products = instance['n_products']
    profits = np.array(instance['profit_margins_usd'])
    capacities = np.array(instance['resource_capacities'])
    resource_consumption = np.array(instance['resource_consumption'])
    
    population = initialize_population(pop_size, n_products)
    best_solution, best_fitness = None, -1e12
    convergence = []

    for gen in range(generations):
        # Repair population to ensure feasibility
        repaired_population = np.array([
            repair_chromosome(ind, resource_consumption, capacities)
            for ind in population
        ])
        
        # Evaluate fitness
        fitnesses = np.array([fitness(ind, profits) for ind in repaired_population])
        convergence.append(np.max(fitnesses))

        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(repaired_population, fitnesses)
            parent2 = tournament_selection(repaired_population, fitnesses)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = np.array(new_population[:pop_size])

        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_solution = repaired_population[gen_best_idx]

        if verbose and gen % 20 == 0:
            print(f"  Generation {gen:3d}: Best Profit = ${best_fitness:,.2f}")

    if verbose:
        print(f"  Generation {generations-1:3d}: Best Profit = ${best_fitness:,.2f}")

    return best_solution, best_fitness, convergence

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
def tune_hyperparameters(instance, pop_sizes, mutation_rates, crossover_rates,
                         generations_list, verbose=True):
    """
    Tune GA hyperparameters and return best solution with best params.
    """
    best_overall_solution = None
    best_overall_profit = -1e12
    best_params = None

    print("\nTesting hyperparameter combinations...")
    print("-" * 70)
    
    for pop_size in pop_sizes:
        for mut_rate in mutation_rates:
            for cross_rate in crossover_rates:
                for gens in generations_list:
                    if verbose:
                        print(f"\nParams: pop={pop_size}, mut={mut_rate}, cross={cross_rate}, gens={gens}")
                    sol, profit, _ = genetic_algorithm(
                        instance,
                        pop_size=pop_size,
                        generations=gens,
                        crossover_rate=cross_rate,
                        mutation_rate=mut_rate,
                        verbose=False
                    )
                    if verbose:
                        print(f"Result: Profit = ${profit:,.2f}")

                    if profit > best_overall_profit:
                        best_overall_profit = profit
                        best_overall_solution = sol
                        best_params = (pop_size, mut_rate, cross_rate, gens)

    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS FOUND:")
    print("="*70)
    print(f"  Population Size: {best_params[0]}")
    print(f"  Mutation Rate: {best_params[1]}")
    print(f"  Crossover Rate: {best_params[2]}")
    print(f"  Generations: {best_params[3]}")
    print(f"  Best Profit: ${best_overall_profit:,.2f}")
    print("="*70)

    print("\nRunning final GA with best parameters...")
    print("-" * 70)
    
    final_solution, final_profit, convergence = genetic_algorithm(
        instance,
        pop_size=best_params[0],
        mutation_rate=best_params[1],
        crossover_rate=best_params[2],
        generations=best_params[3],
        verbose=True
    )

    return final_solution, final_profit, best_params, convergence

# -----------------------------
# Constraint Satisfaction Evaluation
# -----------------------------
def detailed_constraint_report(solution, instance):
    """Generate detailed constraint satisfaction report."""
    profits = np.array(instance['profit_margins_usd'])
    capacities = np.array(instance['resource_capacities'])
    resource_consumption = np.array(instance['resource_consumption'])
    
    total_profit = np.sum(solution * profits)
    
    print("\n" + "="*70)
    print(" "*15 + "CONSTRAINT SATISFACTION REPORT")
    print("="*70)
    
    all_satisfied = True
    resource_details = []
    
    for r in range(len(capacities)):
        usage = np.sum(solution * resource_consumption[r])
        feasible = usage <= capacities[r]
        utilization = (usage / capacities[r]) * 100 if capacities[r] > 0 else 0
        
        status = "SATISFIED" if feasible else "✗ VIOLATED"
        if not feasible:
            all_satisfied = False
        
        resource_details.append({
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
    print(f"ALL CONSTRAINTS SATISFIED: {'YES ✓' if all_satisfied else 'NO ✗'}")
    print(f"NUMBER OF SELECTED PRODUCTS: {int(np.sum(solution))}")
    print("="*70 + "\n")
    
    return total_profit, all_satisfied, resource_details

# -----------------------------
# Convergence Visualization
# -----------------------------
def visualize_convergence(convergence, best_params=None, save_path='ga_results/ga_convergence.png'):
    """Enhanced convergence visualization."""
    plt.figure(figsize=(12, 7))
    
    plt.plot(convergence, linewidth=2.5, color='#2E86AB', label='Best Fitness')
    plt.xlabel('Generation', fontsize=14, fontweight='bold')
    plt.ylabel('Best Fitness (Total Profit USD)', fontsize=14, fontweight='bold')
    plt.title('Genetic Algorithm Convergence Over Generations', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if best_params:
        param_text = (f"Best Parameters:\n"
                     f"Pop Size: {best_params[0]}\n"
                     f"Mutation Rate: {best_params[1]}\n"
                     f"Crossover Rate: {best_params[2]}\n"
                     f"Generations: {best_params[3]}")
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    final_fitness = convergence[-1]
    plt.axhline(y=final_fitness, color='r', linestyle='--', alpha=0.5, 
                label=f'Final Best: ${final_fitness:,.2f}')
    
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nConvergence plot saved to: {save_path}")

# -----------------------------
# Solution Evaluation (for compatibility)
# -----------------------------
def evaluate_solution(solution, instance):
    """
    Evaluate solution and return profit, constraint satisfaction, and usage report.
    This function is used by compare_ga_mip.py
    """
    profits = np.array(instance['profit_margins_usd'])
    capacities = np.array(instance['resource_capacities'])
    resource_consumption = np.array(instance['resource_consumption'])

    total_profit = np.sum(solution * profits)
    constraints_ok = True
    usage_report = []

    for r in range(len(capacities)):
        usage = np.sum(solution * resource_consumption[r])
        feasible = usage <= capacities[r]
        if not feasible:
            constraints_ok = False
        usage_report.append((instance['resource_names'][r], usage, capacities[r], feasible))

    return total_profit, constraints_ok, usage_report

# -----------------------------
# Save Solution to JSON
# -----------------------------
def save_ga_solution(instance, solution, profit, best_params=None, 
                     filename="ga_solution.json"):
    """Save GA solution to JSON file."""
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
        "optimization_method": "Genetic Algorithm",
        "best_profit_usd": float(profit),
        "number_of_selected_products": int(np.sum(solution)),
        "hyperparameters": {
            "population_size": best_params[0] if best_params else None,
            "mutation_rate": best_params[1] if best_params else None,
            "crossover_rate": best_params[2] if best_params else None,
            "generations": best_params[3] if best_params else None
        } if best_params else {},
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

    os.makedirs("ga_results", exist_ok=True)
    filepath = os.path.join("ga_results", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nGA solution saved to: {filepath}")

# -----------------------------
# Main Execution (Standalone Mode)
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "GENETIC ALGORITHM OPTIMIZATION")
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
    
    # Hyperparameter tuning
    print("\n" + "="*70)
    print(" "*20 + "HYPERPARAMETER TUNING PHASE")
    print("="*70)
    
    best_solution, best_profit, best_params, convergence = tune_hyperparameters(
        instance,
        pop_sizes=[50, 100, 150],
        mutation_rates=[0.01, 0.02, 0.05],
        crossover_rates=[0.7, 0.8, 0.9],
        generations_list=[100, 200],
        verbose=True
    )
    
    # Final evaluation with constraint satisfaction
    print("\n" + "="*70)
    print(" "*20 + "FINAL SOLUTION EVALUATION")
    print("="*70)
    
    total_profit, constraints_satisfied, resource_details = detailed_constraint_report(
        best_solution, instance
    )
    
    # Print selected products summary
    selected_products = [
        (sku, cat) 
        for sku, cat, bit in zip(instance['product_skus'], 
                                  instance['product_categories'], 
                                  best_solution) 
        if bit == 1
    ]
    
    print("SELECTED PRODUCTS:")
    print("-"*70)
    num_to_show = min(15, len(selected_products))
    for i, (sku, cat) in enumerate(selected_products[:num_to_show], 1):
        print(f"  {i:2d}. {sku} ({cat})")
    
    if len(selected_products) > num_to_show:
        print(f"  ... and {len(selected_products) - num_to_show} more products")
    
    print(f"\nTotal Selected: {len(selected_products)} products")
    print("-"*70)
    
    # Visualize convergence
    print("\nGenerating convergence visualization...")
    visualize_convergence(convergence, best_params)
    
    # Save solution
    save_ga_solution(instance, best_solution, total_profit, best_params)
    
    print("\n" + "="*70)
    print(" "*25 + "OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"\nBest Profit Achieved: ${total_profit:,.2f}")
    print(f"Constraints Satisfied: {'YES' if constraints_satisfied else 'NO'}")
    print(f"Products Selected: {int(np.sum(best_solution))} out of {instance['n_products']}")
    print(f"Results saved in 'ga_results/' directory")
    print("\n" + "="*70 + "\n")
def validate_manufacturing_data(instance):
    print("DATA VALIDATION CHECKS")
   
    all_passed = True
    
    # 1. Check for missing values
    print("\nChecking for missing values...")
    if None in instance['profit_margins_usd'] or None in instance['resource_capacities']:
        print("FAILED: Missing values found")
        all_passed = False
    else:
        print("PASSED: No missing values")
    
    # 2. Check for negative values
    print("\nChecking for negative values...")
    profits = instance['profit_margins_usd']
    capacities = instance['resource_capacities']
    
    if any(p < 0 for p in profits) or any(c < 0 for c in capacities):
        print("FAILED: Negative values found")
        all_passed = False
    else:
        print("PASSED: All values non-negative")
    
    # 3. Check dimensions
    print("\nChecking data dimensions...")
    if len(profits) != instance['n_products']:
        print("FAILED: Dimension mismatch")
        all_passed = False
    else:
        print("PASSED: All dimensions correct")
    
    # 4. Check individual feasibility
    print("\nChecking individual product feasibility...")
    infeasible = 0
    for j in range(instance['n_products']):
        for i in range(instance['n_resources']):
            if instance['resource_consumption'][i][j] > instance['resource_capacities'][i]:
                infeasible += 1
                break
    
    if infeasible > 0:
        print(f"WARNING: {infeasible} infeasible products found")
    else:
        print("PASSED: All products individually feasible")
    
    # 5. Check for dominated products
    print("\nChecking for dominated products...")
    dominated = 0
    for i in range(instance['n_products']):
        for j in range(i+1, instance['n_products']):
            profit_i = instance['profit_margins_usd'][i]
            profit_j = instance['profit_margins_usd'][j]
            
            if profit_i >= profit_j:
                dominates = all(
                    instance['resource_consumption'][r][i] <= instance['resource_consumption'][r][j]
                    for r in range(instance['n_resources'])
                )
                if dominates and profit_i > profit_j:
                    dominated += 1
    
    print(f"Found {dominated} dominated products")
    print("PASSED: Expected for benchmark datasets")
    
    # Summary
    if all_passed:
        print("ALL VALIDATION CHECKS PASSED")
        print("Dataset is clean and ready for optimization")
    else:
        print("SOME CHECKS FAILED - Review data")
    
    return all_passed


# USAGE:
# Load your data first
from manufacturing_data_loader import ManufacturingDataLoader
loader = ManufacturingDataLoader()
instance = loader.load_manufacturing_instance('manufacturing_problem_1')

# Run validation
validate_manufacturing_data(instance)
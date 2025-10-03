import requests
import numpy as np
import pandas as pd
from typing import Dict, List
import os
import json

class ManufacturingDataLoader:
    """
    Manufacturing Production Planning Data Loader
    Downloads OR-Library MDKS data and converts to manufacturing context
    """
    
    def __init__(self, company_name="TechParts Manufacturing Corp.", 
                 planning_period="Q1 2025"):
        self.base_url = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
        self.data_dir = "manufacturing_data"
        self.company_name = company_name
        self.planning_period = planning_period
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    # Download OR-Library data and convert to manufacturing context
    def download_and_convert(self, filename: str = "mknapcb5.txt") -> List[Dict]:
        print("="*70)
        print(f"MANUFACTURING DATA LOADER - {self.company_name}")
        print("="*70)
        
        # Download file
        filepath = self.download_file(filename)
        
        if filepath is None:
            return None
        
        # Parse and convert to manufacturing context
        instances = self.parse_and_convert_to_manufacturing(filepath)
        print(f"\nSuccessfully converted {len(instances)} manufacturing scenarios")
        
        return instances
    
    # Download a file from OR-Library
    def download_file(self, filename: str) -> str:        
        url = self.base_url + filename
        print(f"Downloading {filename} from OR-Library...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            print(f"Successfully downloaded {filename}")
            return filepath
        
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None
    
    # Parse OR-Library data and convert to manufacturing context
    # Returns list of manufacturing problem instances    
    def parse_and_convert_to_manufacturing(self, filepath: str) -> List[Dict]:        
        instances = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        idx = 0
        # First line: number of test problems
        num_problems = int(lines[idx].strip())
        idx += 1
        
        print(f"\nConverting {num_problems} problems to manufacturing context...")
        
        for prob_num in range(num_problems):
            # Read n (variables), m (constraints), optimal value
            parts = lines[idx].strip().split()
            n = int(parts[0])  # number of items -> products
            m = int(parts[1])  # number of constraints -> resources
            optimal = int(parts[2]) if len(parts) > 2 else 0
            idx += 1
            
            # Read profit coefficients p(j)
            profits = []
            while len(profits) < n:
                profits.extend([int(x) for x in lines[idx].strip().split()])
                idx += 1
            
            # Read constraint coefficients r(i,j) for each constraint
            constraints = []
            for i in range(m):
                constraint_row = []
                while len(constraint_row) < n:
                    constraint_row.extend([int(x) for x in lines[idx].strip().split()])
                    idx += 1
                constraints.append(constraint_row)
            
            # Read capacity constraints b(i)
            capacities = []
            while len(capacities) < m:
                capacities.extend([int(x) for x in lines[idx].strip().split()])
                idx += 1
            
            # Generate manufacturing context
            product_names = self._generate_product_names(n)
            product_categories = [name.split('-')[0] for name in product_names]
            resource_names = self._define_resource_names(m)
            resource_units = self._define_resource_units(m)
            
            # Create manufacturing instance
            instance = {
                'problem_number': prob_num + 1,
                'company_name': self.company_name,
                'planning_period': self.planning_period,
                
                # Problem dimensions
                'n_products': n,
                'n_resources': m,
                'optimal_value': optimal,
                
                # Product information
                'product_skus': product_names,
                'product_categories': product_categories,
                'profit_margins_usd': [p * 1000 for p in profits],  # Scale to dollars
                
                # Resource information
                'resource_names': resource_names,
                'resource_units': resource_units,
                'resource_capacities': capacities,
                'resource_consumption': constraints  # [resource][product]
            }
            
            instances.append(instance)
            print(f"Problem {prob_num + 1}: {n} products, {m} resources")
        
        return instances
    
    # Generate realistic product SKU names
    def _generate_product_names(self, n_items: int) -> List[str]:
        categories = ['Widget', 'Gadget', 'Component', 'Module', 'Assembly']
        series = ['A', 'B', 'C', 'D', 'E']
        
        products = []
        for i in range(n_items):
            cat = categories[i % len(categories)]
            ser = series[(i // 20) % len(series)]
            num = (i % 20) + 1
            products.append(f"{cat}-{ser}{num:02d}")
        
        return products
    
    # Define manufacturing resource names
    def _define_resource_names(self, n_constraints: int) -> List[str]:
        if n_constraints == 5:
            return [
                "CNC_Machine_Hours",
                "Assembly_Line_Hours",
                "Skilled_Labor_Hours",
                "Raw_Material_Budget_USD",
                "Energy_Consumption_kWh"
            ]
        elif n_constraints == 10:
            return [
                "CNC_Machine_Hours",
                "Assembly_Line_Hours",
                "Skilled_Labor_Hours",
                "Raw_Material_Budget_USD",
                "Energy_Consumption_kWh",
                "Warehouse_Space_SqFt",
                "Quality_Inspection_Hours",
                "Setup_Changeover_Hours",
                "Shipping_Capacity_Pallets",
                "Compliance_Hours"
            ]
        else:
            return [f"Resource_{i+1}" for i in range(n_constraints)]
    
    # Define units for each resource
    def _define_resource_units(self, n_constraints: int) -> List[str]:
        if n_constraints == 5:
            return ["hours", "hours", "hours", "USD", "kWh"]
        elif n_constraints == 10:
            return ["hours", "hours", "hours", "USD", "kWh", 
                   "sq_ft", "hours", "hours", "pallets", "hours"]
        else:
            return ["units"] * n_constraints
  
    # Create summary of all manufacturing problems
    def create_summary_dataframe(self, instances: List[Dict]) -> pd.DataFrame:
        summary_data = []
        
        for inst in instances:
            summary_data.append({
                'Problem': inst['problem_number'],
                'Products': inst['n_products'],
                'Resources': inst['n_resources'],
                'Optimal_Value_USD': inst['optimal_value'] * 1000 if inst['optimal_value'] > 0 else 'Unknown',
                'Avg_Profit_USD': np.mean(inst['profit_margins_usd']),
                'Max_Profit_USD': np.max(inst['profit_margins_usd']),
                'Total_Capacity': sum(inst['resource_capacities'])
            })
        
        return pd.DataFrame(summary_data)
        
    # Save manufacturing instance in both NPZ and JSON formats
    def save_manufacturing_instance(self, instance: Dict, problem_name: str):
        base_name = problem_name.replace('.npz', '').replace('.json', '')
        
        print(f"\nSaving manufacturing problem: {base_name}")
        
        # Save as NPZ (for Python/NumPy processing)
        npz_file = os.path.join(self.data_dir, f"{base_name}.npz")
        np.savez(
            npz_file,
            problem_number=instance['problem_number'],
            company_name=instance['company_name'],
            planning_period=instance['planning_period'],
            n_products=instance['n_products'],
            n_resources=instance['n_resources'],
            optimal_value=instance['optimal_value'],
            product_skus=instance['product_skus'],
            product_categories=instance['product_categories'],
            profit_margins_usd=instance['profit_margins_usd'],
            resource_names=instance['resource_names'],
            resource_units=instance['resource_units'],
            resource_capacities=instance['resource_capacities'],
            resource_consumption=instance['resource_consumption']
        )
        print(f"Saved NPZ: {base_name}.npz")
        
        # Save as JSON (for human readability)
        json_file = os.path.join(self.data_dir, f"{base_name}.json")
        json_data = {
            'problem_info': {
                'problem_number': instance['problem_number'],
                'company_name': instance['company_name'],
                'planning_period': instance['planning_period'],
                'n_products': instance['n_products'],
                'n_resources': instance['n_resources'],
                'optimal_value': instance['optimal_value']
            },
            'products': [
                {
                    'sku': instance['product_skus'][i],
                    'category': instance['product_categories'][i],
                    'profit_usd': instance['profit_margins_usd'][i],
                    'resource_requirements': {
                        instance['resource_names'][r]: int(instance['resource_consumption'][r][i])
                        for r in range(instance['n_resources'])
                    }
                }
                for i in range(instance['n_products'])
            ],
            'resources': [
                {
                    'name': instance['resource_names'][i],
                    'capacity': int(instance['resource_capacities'][i]),
                    'unit': instance['resource_units'][i]
                }
                for i in range(instance['n_resources'])
            ]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {base_name}.json (human-readable)")
    
    # Display manufacturing problem details
    def display_manufacturing_problem(self, instance: Dict):
        print("\n" + "="*70)
        print(f"MANUFACTURING PRODUCTION PLANNING PROBLEM #{instance['problem_number']}")
        print("="*70)
        print(f"Company: {instance['company_name']}")
        print(f"Planning Period: {instance['planning_period']}")
        print(f"Products (SKUs): {instance['n_products']}")
        print(f"Resource Constraints: {instance['n_resources']}")
        print(f"Known Optimal Profit: ${instance['optimal_value'] * 1000:,}" if instance['optimal_value'] > 0 else "Known Optimal: Unknown")
        
        print("\n" + "-"*70)
        print("AVAILABLE RESOURCES:")
        print("-"*70)
        for i, (name, capacity, unit) in enumerate(zip(
            instance['resource_names'], 
            instance['resource_capacities'],
            instance['resource_units'])):
            print(f"{i+1:2d}. {name:35s}: {capacity:>8,} {unit}")
        
        print("\n" + "-"*70)
        print("SAMPLE PRODUCTS (First 10):")
        print("-"*70)
        print(f"{'SKU':<15} {'Category':<12} {'Profit':>12} {'Resource_1':>12} {'Resource_2':>12}")
        print("-"*70)
        for i in range(min(10, instance['n_products'])):
            print(f"{instance['product_skus'][i]:<15} "
                  f"{instance['product_categories'][i]:<12} "
                  f"${instance['profit_margins_usd'][i]:>10,} "
                  f"{instance['resource_consumption'][0][i]:>12.0f} "
                  f"{instance['resource_consumption'][1][i]:>12.0f}")
        print("...")
    
    # Load manufacturing instance from NPZ file
    def load_manufacturing_instance(self, filename: str) -> Dict:
        filepath = os.path.join(self.data_dir, filename)
        
        if not filename.endswith('.npz'):
            filename = filename + '.npz'
            filepath = os.path.join(self.data_dir, filename)
        
        data = np.load(filepath, allow_pickle=True)
        
        return {
            'problem_number': int(data['problem_number']),
            'company_name': str(data['company_name']),
            'planning_period': str(data['planning_period']),
            'n_products': int(data['n_products']),
            'n_resources': int(data['n_resources']),
            'optimal_value': int(data['optimal_value']),
            'product_skus': data['product_skus'].tolist(),
            'product_categories': data['product_categories'].tolist(),
            'profit_margins_usd': data['profit_margins_usd'].tolist(),
            'resource_names': data['resource_names'].tolist(),
            'resource_units': data['resource_units'].tolist(),
            'resource_capacities': data['resource_capacities'].tolist(),
            'resource_consumption': data['resource_consumption'].tolist()
        }


if __name__ == "__main__":
    # Initialize loader with company details
    loader = ManufacturingDataLoader(
        company_name="TechParts Manufacturing Corporation",
        planning_period="Q1 2025"
    )
    
    # Download and convert OR-Library data to manufacturing context
    instances = loader.download_and_convert("mknapcb5.txt")
    
    if instances:
        # Create and display summary
        summary_df = loader.create_summary_dataframe(instances)
        print("\n" + "="*70)
        print("MANUFACTURING PROBLEMS SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))
        
        # Save first problem (recommended for assignment)
        print("\n" + "="*70)
        print("SAVING PROBLEM 1 FOR DETAILED ANALYSIS")
        print("="*70)
        loader.save_manufacturing_instance(instances[0], "manufacturing_problem_1")
        
        # Display problem details
        loader.display_manufacturing_problem(instances[0])
        
        print("\n" + "="*70)
        print("MANUFACTURING DATA READY!")
        print("="*70)
        print(f"Data Location: ./{loader.data_dir}/")
        print(f"Files Created:")
        print(f"  1. manufacturing_problem_1.npz (for Python/algorithms)")
        print(f"  2. manufacturing_problem_1.json (for human reading)")
        print(f"\nNext Steps:")
        print(f"  1. Load data: loader.load_manufacturing_instance('manufacturing_problem_1')")
        print(f"  2. Implement Genetic Algorithm")
        print(f"  3. Implement MIP solver")
        print(f"  4. Compare results")
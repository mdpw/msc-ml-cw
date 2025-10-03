import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from manufacturing_data_loader import ManufacturingDataLoader

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ManufacturingEDA:
    def __init__(self, instance):
        self.instance = instance
        self.n_products = instance['n_products']
        self.n_resources = instance['n_resources']
        
        # Create DataFrame for products
        self.products_df = self._create_products_dataframe()
        
        # Create DataFrame for resources
        self.resources_df = self._create_resources_dataframe()
        
    def _create_products_dataframe(self):
        data = []
        for i in range(self.n_products):
            row = {
                'SKU': self.instance['product_skus'][i],
                'Category': self.instance['product_categories'][i],
                'Profit_USD': self.instance['profit_margins_usd'][i]
            }
            
            # Add resource consumption for each resource
            for r in range(self.n_resources):
                resource_name = self.instance['resource_names'][r]
                row[f'Consumes_{resource_name}'] = self.instance['resource_consumption'][r][i]
            
            # Calculate total resource consumption
            row['Total_Resource_Consumption'] = sum(
                self.instance['resource_consumption'][r][i] 
                for r in range(self.n_resources)
            )
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _create_resources_dataframe(self):
        data = []
        for i in range(self.n_resources):
            data.append({
                'Resource': self.instance['resource_names'][i],
                'Capacity': self.instance['resource_capacities'][i],
                'Unit': self.instance['resource_units'][i]
            })
        return pd.DataFrame(data)
    
    def run_complete_eda(self):
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print(f"{self.instance['company_name']} - {self.instance['planning_period']}")
        
        # 1. Basic Statistics
        self.basic_statistics()

        # 2. Resource Analysis
        self.resource_analysis()
            
        # 3. Efficiency Analysis
        self.efficiency_analysis()        

        # 4. Feasibility Analysis
        self.feasibility_analysis()
        
        print("\n" + "="*80)
        print("EDA COMPLETE - Ready for visualization and modeling")
        print("="*80)
    
    def basic_statistics(self):       
        print("1. BASIC STATISTICS")       
        print("\nProblem Dimensions:")
        print(f"Number of Products: {self.n_products}")
        print(f"Number of Resources: {self.n_resources}")
        print(f"Solution Space Size: 2^{self.n_products} = {2**self.n_products:.2e} combinations")
        print(f"Known Optimal Value: ${self.instance['optimal_value']*1000:,.0f}")
        
        print("\nProfit Statistics (USD):")
        profit_stats = self.products_df['Profit_USD'].describe()
        print(f"Mean:     ${profit_stats['mean']:>12,.2f}")
        print(f"Median:   ${profit_stats['50%']:>12,.2f}")
        print(f"Std Dev:  ${profit_stats['std']:>12,.2f}")
        print(f"Min:      ${profit_stats['min']:>12,.2f}")
        print(f"Max:      ${profit_stats['max']:>12,.2f}")
        print(f"Range:    ${profit_stats['max'] - profit_stats['min']:>12,.2f}")
        
        # Coefficient of Variation
        cv = (profit_stats['std'] / profit_stats['mean']) * 100
        print(f"CV:       {cv:>12.2f}% (Profit variability)")
        
        print("\nResource Consumption Statistics:")
        for r in range(self.n_resources):
            col_name = f"Consumes_{self.instance['resource_names'][r]}"
            stats_r = self.products_df[col_name].describe()
            print(f"\n{self.instance['resource_names'][r]}:")
            print(f"Mean: {stats_r['mean']:>10.2f} | Std: {stats_r['std']:>10.2f}")
            print(f"Min:  {stats_r['min']:>10.2f} | Max: {stats_r['max']:>10.2f}")
     
    def resource_analysis(self):    
        print("RESOURCE CAPACITY ANALYSIS")        
        print("\nResource Capacities:")
        total_capacity = 0
        
        for i in range(self.n_resources):
            capacity = self.instance['resource_capacities'][i]
            unit = self.instance['resource_units'][i]
            name = self.instance['resource_names'][i]
            
            print(f"{name:35s}: {capacity:>10,.0f} {unit}")
            total_capacity += capacity
        
        print(f"{'Total Capacity (sum)':35s}: {total_capacity:>10,.0f}")
        
        # Calculate demand if all products selected
        print("\nResource Demand Analysis (if ALL products selected):")   
        
        for i in range(self.n_resources):
            name = self.instance['resource_names'][i]
            capacity = self.instance['resource_capacities'][i]
            
            total_demand = sum(self.instance['resource_consumption'][i])
            avg_demand = np.mean(self.instance['resource_consumption'][i])
            
            utilization = (total_demand / capacity) * 100
            
            print(f"\n{name}:")
            print(f"Total Demand:    {total_demand:>10,.0f}")
            print(f"Capacity:        {capacity:>10,.0f}")
            print(f"Utilization:     {utilization:>10,.1f}%", end="")
            
            if utilization > 100:
                print("OVERSUBSCRIBED - Constraint is TIGHT")
            else:
                print("Sufficient capacity")
            
            print(f"Avg per product: {avg_demand:>10,.2f}")
    
    def efficiency_analysis(self):
        print("EFFICIENCY ANALYSIS (Profit per Unit Resource)")
        
        # Calculate efficiency ratios
        efficiency_data = []
        
        for i in range(self.n_products):
            for r in range(self.n_resources):
                consumption = self.instance['resource_consumption'][r][i]
                profit = self.instance['profit_margins_usd'][i]
                
                if consumption > 0:
                    efficiency = profit / consumption
                    efficiency_data.append({
                        'Product': self.instance['product_skus'][i],
                        'Resource': self.instance['resource_names'][r],
                        'Efficiency': efficiency,
                        'Profit': profit,
                        'Consumption': consumption
                    })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        
        print("\nAverage Efficiency by Resource:")        
        
        for resource in efficiency_df['Resource'].unique():
            resource_eff = efficiency_df[efficiency_df['Resource'] == resource]['Efficiency']
            print(f"{resource:35s}: ${resource_eff.mean():>10,.2f} profit/unit")
        
        # Find most efficient products
        print("\nTop 5 Most Efficient Products (Overall):")        
        
        # Calculate average efficiency across all resources
        product_efficiency = efficiency_df.groupby('Product')['Efficiency'].mean().sort_values(ascending=False)
        
        for i, (product, eff) in enumerate(product_efficiency.head(5).items(), 1):
            profit = self.products_df[self.products_df['SKU'] == product]['Profit_USD'].values[0]
            print(f"{i}. {product:15s} - ${eff:>10,.2f} avg efficiency | ${profit:>10,.0f} profit")
      
    def feasibility_analysis(self):
        print("FEASIBILITY ANALYSIS")
        
        # Check if individual products are feasible
        print("\nIndividual Product Feasibility:")
        
        infeasible_products = []
        
        for i in range(self.n_products):
            is_feasible = True
            violating_resources = []
            
            for r in range(self.n_resources):
                consumption = self.instance['resource_consumption'][r][i]
                capacity = self.instance['resource_capacities'][r]
                
                if consumption > capacity:
                    is_feasible = False
                    violating_resources.append(self.instance['resource_names'][r])
            
            if not is_feasible:
                infeasible_products.append({
                    'SKU': self.instance['product_skus'][i],
                    'Violations': violating_resources
                })
        
        if len(infeasible_products) == 0:
            print("All products are individually feasible")
        else:
            print(f"{len(infeasible_products)} products violate capacity constraints:")
            for prod in infeasible_products[:5]:
                print(f"{prod['SKU']:15s} violates: {', '.join(prod['Violations'])}")
        
        # Estimate maximum number of products that can be selected
        print("\nCapacity Utilization Estimates:")
        
        # Greedy estimation: sort by profit, select until infeasible
        sorted_indices = np.argsort([-p for p in self.instance['profit_margins_usd']])
        
        selected = 0
        remaining_capacity = list(self.instance['resource_capacities'])
        
        for idx in sorted_indices:
            can_add = True
            for r in range(self.n_resources):
                if self.instance['resource_consumption'][r][idx] > remaining_capacity[r]:
                    can_add = False
                    break
            
            if can_add:
                selected += 1
                for r in range(self.n_resources):
                    remaining_capacity[r] -= self.instance['resource_consumption'][r][idx]
        
        print(f"Greedy upper bound: ~{selected} products can be selected")
        print(f"This is {(selected/self.n_products)*100:.1f}% of total products")
        
        print("\nRemaining Capacity after greedy selection:")
        for r in range(self.n_resources):
            original = self.instance['resource_capacities'][r]
            remaining = remaining_capacity[r]
            used_pct = ((original - remaining) / original) * 100
            
            print(f"      {self.instance['resource_names'][r]:30s}: "
                  f"{remaining:>8,.0f} / {original:>8,.0f} "
                  f"({used_pct:>5.1f}% used)")
    
    def generate_summary_report(self):
        print("EXECUTIVE SUMMARY - KEY INSIGHTS")
        
        # Key metrics
        profit_mean = self.products_df['Profit_USD'].mean()
        profit_std = self.products_df['Profit_USD'].std()
        profit_cv = (profit_std / profit_mean) * 100
        
        print(f"\nDataset Overview:")
        print(f"{self.n_products} products across {len(self.products_df['Category'].unique())} categories")
        print(f"{self.n_resources} resource constraints")
        print(f"Solution space: {2**self.n_products:.2e} combinations")
        
        print(f"\nProfit Characteristics:")
        print(f"Average profit: ${profit_mean:,.0f} with {profit_cv:.1f}% variability")
        print(f"Profit range: ${self.products_df['Profit_USD'].min():,.0f} to ${self.products_df['Profit_USD'].max():,.0f}")
        
        # Find bottleneck resource
        resource_utilization = []
        for r in range(self.n_resources):
            total_demand = sum(self.instance['resource_consumption'][r])
            capacity = self.instance['resource_capacities'][r]
            utilization = (total_demand / capacity) * 100
            resource_utilization.append((self.instance['resource_names'][r], utilization))
        
        bottleneck = max(resource_utilization, key=lambda x: x[1])
        
        print(f"\nCritical Insights:")
        print(f"Bottleneck resource: {bottleneck[0]} ({bottleneck[1]:.0f}% utilization if all selected)")
        print(f"Optimization complexity: NP-Hard (requires heuristic approaches)")
        print(f"Data quality: Complete, no missing values")    
       

if __name__ == "__main__":
    # Load the manufacturing problem
    # Make sure ManufacturingDataLoader.py is in the same directory or update the import path accordingly

    
    loader = ManufacturingDataLoader()
    
    # Load saved instance
    instance = loader.load_manufacturing_instance('manufacturing_problem_1')
    
    # Run EDA
    eda = ManufacturingEDA(instance)
    eda.run_complete_eda()
    
    # Generate summary
    eda.generate_summary_report()
    
    # Access the dataframes for further analysis or visualization
    products_df = eda.products_df
    resources_df = eda.resources_df
    
    print("\nDataFrames available for further analysis:")
    print("eda.products_df - Complete product information")
    print("eda.resources_df - Resource information")
    print("\nUse these for custom visualizations and deep-dive analysis!")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from manufacturing_data_loader import ManufacturingDataLoader

class ManufacturingVisualization:
    """
    Comprehensive visualization suite for Manufacturing EDA
    Creates publication-quality plots for analysis
    """
    
    def __init__(self, eda_instance):
        self.eda = eda_instance
        self.products_df = eda_instance.products_df
        self.instance = eda_instance.instance
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
        
    def plot_all_visualizations(self, save_figs=False):
        """Generate all visualization plots"""
        print("Generating visualizations...")
        
        # Create figure directory if saving
        if save_figs:
            import os
            if not os.path.exists('eda_figures'):
                os.makedirs('eda_figures')
        
        # 1. Profit distribution
        self.plot_profit_distribution(save_figs)
        
# 6. Resource capacity vs demand
        self.plot_capacity_analysis(save_figs)

        # 4. Correlation heatmap
        self.plot_correlation_heatmap(save_figs)
        
            # 5. Efficiency analysis
        self.plot_efficiency_analysis(save_figs)
    
        self.create_summary_dashboard(save=True)
        

        
        print("✅ All visualizations complete!")
        plt.show()
    
    def plot_profit_distribution(self, save=False):
        """1. Profit Distribution Analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Profit Distribution Analysis', fontsize=16, fontweight='bold')
        
        profit_data = self.products_df['Profit_USD']
        
        # Histogram
        axes[0, 0].hist(profit_data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(profit_data.mean(), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: ${profit_data.mean():,.0f}')
        axes[0, 0].axvline(profit_data.median(), color='green', linestyle='--', 
                           linewidth=2, label=f'Median: ${profit_data.median():,.0f}')
        axes[0, 0].set_xlabel('Profit (USD)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Histogram of Product Profits', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        bp = axes[0, 1].boxplot(profit_data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        axes[0, 1].set_ylabel('Profit (USD)', fontsize=12)
        axes[0, 1].set_title('Box Plot - Outlier Detection', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot for normality
        stats.probplot(profit_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # KDE plot
        profit_data.plot(kind='density', ax=axes[1, 1], color='purple', linewidth=2)
        axes[1, 1].fill_between(axes[1, 1].lines[0].get_xdata(), 
                                axes[1, 1].lines[0].get_ydata(), 
                                alpha=0.3, color='purple')
        axes[1, 1].set_xlabel('Profit (USD)', fontsize=12)
        axes[1, 1].set_ylabel('Density', fontsize=12)
        axes[1, 1].set_title('Kernel Density Estimation', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Mean: ${profit_data.mean():,.0f}\nMedian: ${profit_data.median():,.0f}\n"
        stats_text += f"Std: ${profit_data.std():,.0f}\nSkew: {stats.skew(profit_data):.3f}"
        axes[1, 1].text(0.65, 0.95, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if save:
            plt.savefig('eda_figures/01_profit_distribution.png', dpi=300, bbox_inches='tight')
        
    
        """2. Product Category Analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Product Category Analysis', fontsize=16, fontweight='bold')
        
        # Count by category
        category_counts = self.products_df['Category'].value_counts()
        axes[0, 0].bar(category_counts.index, category_counts.values, 
                       color=sns.color_palette("Set2", len(category_counts)))
        axes[0, 0].set_xlabel('Category', fontsize=12)
        axes[0, 0].set_ylabel('Number of Products', fontsize=12)
        axes[0, 0].set_title('Product Count by Category', fontsize=13, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Average profit by category
        category_profit = self.products_df.groupby('Category')['Profit_USD'].mean().sort_values(ascending=False)
        axes[0, 1].barh(category_profit.index, category_profit.values, color='coral')
        axes[0, 1].set_xlabel('Average Profit (USD)', fontsize=12)
        axes[0, 1].set_ylabel('Category', fontsize=12)
        axes[0, 1].set_title('Average Profit by Category', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Box plot by category
        self.products_df.boxplot(column='Profit_USD', by='Category', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Category', fontsize=12)
        axes[1, 0].set_ylabel('Profit (USD)', fontsize=12)
        axes[1, 0].set_title('Profit Distribution by Category', fontsize=13, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45)
        
        # Violin plot
        sns.violinplot(data=self.products_df, x='Category', y='Profit_USD', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Category', fontsize=12)
        axes[1, 1].set_ylabel('Profit (USD)', fontsize=12)
        axes[1, 1].set_title('Profit Density by Category', fontsize=13, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save:
            plt.savefig('eda_figures/02_category_analysis.png', dpi=300, bbox_inches='tight')
    
    def plot_correlation_heatmap(self, save=False):
        """4. Correlation Heatmap"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Get numeric columns
        numeric_cols = ['Profit_USD'] + [col for col in self.products_df.columns 
                                         if col.startswith('Consumes_')]
        
        # Clean column names for display
        display_names = ['Profit'] + [col.replace('Consumes_', '') for col in numeric_cols[1:]]
        
        corr_matrix = self.products_df[numeric_cols].corr()
        corr_matrix.index = display_names
        corr_matrix.columns = display_names
        
        # Full correlation heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=axes[0], cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Complete Correlation Matrix', fontsize=13, fontweight='bold')
        
        # Profit correlation only
        profit_corr = corr_matrix[['Profit']].sort_values('Profit', ascending=False)
        
        colors = ['green' if x > 0 else 'red' for x in profit_corr['Profit'].values]
        axes[1].barh(range(len(profit_corr)), profit_corr['Profit'].values, color=colors, alpha=0.7)
        axes[1].set_yticks(range(len(profit_corr)))
        axes[1].set_yticklabels(profit_corr.index)
        axes[1].set_xlabel('Correlation with Profit', fontsize=12)
        axes[1].set_title('Profit vs Resource Correlations', fontsize=13, fontweight='bold')
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save:
            plt.savefig('eda_figures/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    
    def plot_efficiency_analysis(self, save=False):
        """5. Efficiency Analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Efficiency Analysis (Profit per Unit Resource)', 
                     fontsize=16, fontweight='bold')
        
        n_resources = self.instance['n_resources']
        
        # Calculate efficiency for first 4 resources (or all if less than 4)
        for idx in range(min(4, n_resources)):
            row = idx // 2
            col = idx % 2
            
            resource_name = self.instance['resource_names'][idx]
            col_name = f"Consumes_{resource_name}"
            
            # Calculate efficiency
            efficiency = self.products_df['Profit_USD'] / (self.products_df[col_name] + 1e-6)
            efficiency = efficiency.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Plot
            axes[row, col].hist(efficiency, bins=30, edgecolor='black', 
                               alpha=0.7, color=sns.color_palette("Set2")[idx])
            axes[row, col].axvline(efficiency.mean(), color='red', linestyle='--', 
                                  linewidth=2, label=f'Mean: ${efficiency.mean():,.2f}')
            axes[row, col].set_xlabel('Profit per Unit (USD)', fontsize=11)
            axes[row, col].set_ylabel('Frequency', fontsize=11)
            axes[row, col].set_title(f'Efficiency: {resource_name[:30]}', 
                                    fontsize=12, fontweight='bold')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig('eda_figures/05_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    
    def plot_capacity_analysis(self, save=False):
        """6. Resource Capacity vs Demand Analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Resource Capacity vs Demand Analysis', fontsize=16, fontweight='bold')
        
        n_resources = self.instance['n_resources']
        
        # Prepare data
        capacities = []
        total_demands = []
        avg_demands = []
        resource_names = []
        
        for r in range(n_resources):
            name = self.instance['resource_names'][r]
            capacity = self.instance['resource_capacities'][r]
            total_demand = sum(self.instance['resource_consumption'][r])
            avg_demand = np.mean(self.instance['resource_consumption'][r])
            
            capacities.append(capacity)
            total_demands.append(total_demand)
            avg_demands.append(avg_demand)
            resource_names.append(name[:20])  # Truncate long names
        
        # Plot 1: Capacity vs Total Demand
        x = np.arange(len(resource_names))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, capacities, width, label='Capacity', 
                           color='steelblue', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, total_demands, width, label='Total Demand (if all selected)', 
                           color='coral', alpha=0.8)
        
        axes[0].set_xlabel('Resource', fontsize=12)
        axes[0].set_ylabel('Amount', fontsize=12)
        axes[0].set_title('Capacity vs Total Demand', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(resource_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if capacities[i] > 0:
                pct = (total_demands[i] / capacities[i]) * 100
                height = max(bar1.get_height(), bar2.get_height())
                axes[0].text(x[i], height * 1.05, f'{pct:.0f}%', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Utilization percentage
        utilization = [(td/cap)*100 if cap > 0 else 0 
                      for td, cap in zip(total_demands, capacities)]
        
        colors = ['red' if u > 100 else 'orange' if u > 80 else 'green' 
                 for u in utilization]
        
        bars = axes[1].barh(resource_names, utilization, color=colors, alpha=0.7)
        axes[1].axvline(100, color='red', linestyle='--', linewidth=2, 
                       label='100% Capacity', alpha=0.7)
        axes[1].set_xlabel('Utilization (%)', fontsize=12)
        axes[1].set_title('Resource Utilization (if all products selected)', 
                         fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, utilization)):
            axes[1].text(val + 5, i, f'{val:.1f}%', 
                        va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig('eda_figures/06_capacity_analysis.png', dpi=300, bbox_inches='tight')
    
    def create_summary_dashboard(self, save=False):
        """9. Executive Summary Dashboard"""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('EXECUTIVE SUMMARY DASHBOARD', fontsize=18, fontweight='bold')
        
        # 1. Problem Overview (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        overview_text = f"""
PROBLEM OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━
Company: {self.instance['company_name']}
Period: {self.instance['planning_period']}

Products: {self.instance['n_products']}
Resources: {self.instance['n_resources']}
Solution Space: {2**self.instance['n_products']:.2e}

Optimal Value: ${self.instance['optimal_value']*1000:,.0f}
        """
        ax1.text(0.1, 0.9, overview_text, fontsize=11, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 2. Profit statistics (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        profit_data = self.products_df['Profit_USD']
        ax2.hist(profit_data, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(profit_data.mean(), color='red', linestyle='--', linewidth=2)
        ax2.set_title('Profit Distribution', fontweight='bold')
        ax2.set_xlabel('Profit (USD)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Category breakdown (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        category_counts = self.products_df['Category'].value_counts()
        ax3.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
               startangle=90, colors=sns.color_palette("Set2"))
        ax3.set_title('Product Categories', fontweight='bold')
        
        # 4. Resource utilization (middle row)
        ax4 = fig.add_subplot(gs[1, :])
        
        utilizations = []
        names = []
        for r in range(self.instance['n_resources']):
            total_demand = sum(self.instance['resource_consumption'][r])
            capacity = self.instance['resource_capacities'][r]
            util = (total_demand / capacity) * 100
            utilizations.append(util)
            names.append(self.instance['resource_names'][r][:20])
        
        colors = ['red' if u > 100 else 'orange' if u > 80 else 'green' for u in utilizations]
        bars = ax4.barh(names, utilizations, color=colors, alpha=0.7)
        ax4.axvline(100, color='red', linestyle='--', linewidth=2, label='100% Capacity')
        ax4.set_xlabel('Utilization (%)', fontsize=12)
        ax4.set_title('Resource Utilization (if all products selected)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Top 10 products (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        top_10 = self.products_df.nlargest(10, 'Profit_USD')[['SKU', 'Profit_USD']]
        y_pos = np.arange(len(top_10))
        ax5.barh(y_pos, top_10['Profit_USD'].values, color='gold', alpha=0.8)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(top_10['SKU'].values, fontsize=9)
        ax5.invert_yaxis()
        ax5.set_xlabel('Profit (USD)')
        ax5.set_title('Top 10 Most Profitable Products', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Key insights (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        bottleneck_idx = np.argmax(utilizations)
        bottleneck_name = names[bottleneck_idx]
        bottleneck_util = utilizations[bottleneck_idx]
        
        insights_text = f"""
KEY INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━
Profit Range:
  ${profit_data.min():,.0f} - ${profit_data.max():,.0f}
  
Average Profit:
  ${profit_data.mean():,.0f}
  
Bottleneck Resource:
  {bottleneck_name}
  ({bottleneck_util:.0f}% utilization)
  
Problem Complexity:
  NP-Hard (requires heuristics)
        """
        ax6.text(0.1, 0.9, insights_text, fontsize=10, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # 7. Correlation summary (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        numeric_cols = ['Profit_USD'] + [col for col in self.products_df.columns 
                                         if col.startswith('Consumes_')]
        corr_matrix = self.products_df[numeric_cols].corr()
        profit_corr = corr_matrix['Profit_USD'][1:].values
        resource_names_short = [name[:15] for name in names]
        
        colors = ['green' if x > 0 else 'red' for x in profit_corr]
        ax7.barh(resource_names_short[:len(profit_corr)], profit_corr, color=colors, alpha=0.7)
        ax7.axvline(0, color='black', linewidth=0.8)
        ax7.set_xlabel('Correlation')
        ax7.set_title('Profit-Resource Correlation', fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='x')
        
        if save:
            plt.savefig('eda_figures/09_summary_dashboard.png', dpi=300, bbox_inches='tight')


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__": 
    from manufacturing_eda import ManufacturingEDA
    
    # Load data
    loader = ManufacturingDataLoader()
    instance = loader.load_manufacturing_instance('manufacturing_problem_1')
    
    # Run EDA
    eda = ManufacturingEDA(instance)
    
    # Create visualizations
    viz = ManufacturingVisualization(eda)
    
    # Generate all plots (set save_figs=True to save to disk)
    viz.plot_all_visualizations(save_figs=True)
    
    # Or generate specific plots
    # viz.plot_profit_distribution(save=True)
    # viz.plot_capacity_analysis(save=True)
    # viz.create_summary_dashboard(save=True)
    
    print("\n✅ All visualizations generated!")
    print("📁 Figures saved in: ./eda_figures/")
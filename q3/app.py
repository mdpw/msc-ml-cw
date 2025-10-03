from manufacturing_data_loader import ManufacturingDataLoader
from manufacturing_eda import ManufacturingEDA
from manufacturing_eda_viz import ManufacturingVisualization

# Run everything
loader = ManufacturingDataLoader()
instance = loader.load_manufacturing_instance('manufacturing_problem_1')

eda = ManufacturingEDA(instance)
eda.run_complete_eda()

viz = ManufacturingVisualization(eda)
viz.plot_all_visualizations(save_figs=True)
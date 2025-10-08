from manufacturing_data_loader import ManufacturingDataLoader
from manufacturing_eda import ManufacturingEDA
from manufacturing_eda_viz import ManufacturingVisualization
import yaml

def load_config():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# Run everything
loader = ManufacturingDataLoader()
instance = loader.load_manufacturing_instance(config['data_settings']['problem_instance_name'])

eda = ManufacturingEDA(instance)
eda.run_complete_eda()

viz = ManufacturingVisualization(eda)
viz.plot_all_visualizations(save_figs=True)
import yaml
from src.t_model import SummaryModel

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = SummaryModel(config)
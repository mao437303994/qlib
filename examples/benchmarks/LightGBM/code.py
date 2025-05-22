from os import path
from qlib.workflow.cli import workflow

if __name__ == "__main__":
    path = path.join(path.dirname(__file__), "workflow_config_future_lightgbm_Alpha158.yaml")
    workflow(config_path=path)

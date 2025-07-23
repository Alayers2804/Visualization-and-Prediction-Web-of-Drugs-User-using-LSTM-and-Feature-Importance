from datetime import datetime
from app.util.visualization import Visualizer

# Global dataset path
FILE_PATH = "data/dataset_tat_rehab.xlsx"

# Timestamp used in filenames
def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

visualizer = Visualizer()
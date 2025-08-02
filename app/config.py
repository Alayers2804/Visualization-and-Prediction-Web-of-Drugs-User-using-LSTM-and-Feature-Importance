from datetime import datetime
from app.util.visualization import Visualizer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.model.database import Base

# Global dataset path
FILE_PATH = "data/dataset_tat_rehab.xlsx"
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/db.sqlite3"


# Timestamp used in filenames
def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

visualizer = Visualizer()


engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def init_db():
    Base.metadata.create_all(bind=engine)
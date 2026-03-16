from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
import os
import datetime
import uuid

# Get connection string from environment variable or use local SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./econosim.db")

# For sqlite we need connect_args={"check_same_thread": False}
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    
    # Storing serialized objects for dynamic configuration/results
    config = Column(JSON, nullable=False)
    summary = Column(JSON, nullable=False)
    periods = Column(JSON, nullable=False)
    aggregate = Column(JSON, nullable=True)

# Create tables if they don't exist
def init_db():
    Base.metadata.create_all(bind=engine)

# Dependency for FastAPI endpoints
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

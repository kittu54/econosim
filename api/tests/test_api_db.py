import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from api.database import Base, get_db

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_econosim.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def run_around_tests():
    # Setup - drop and recreate tables before each test
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    # Teardown
    Base.metadata.drop_all(bind=engine)

def test_save_run():
    response = client.post(
        "/api/runs",
        json={
            "name": "Test Run",
            "config": {"seed": 42, "num_periods": 10},
            "summary": {"gdp": 100},
            "periods": [{"period": 0, "gdp": 100}, {"period": 1, "gdp": 105}],
            "aggregate": None
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["name"] == "Test Run"
    assert "id" in data
    assert "periods" not in data # Periods excluded from basic response

def test_list_runs():
    # Create a couple of runs
    client.post("/api/runs", json={
        "name": "Run 1", "config": {}, "summary": {}, "periods": []
    })
    client.post("/api/runs", json={
        "name": "Run 2", "config": {}, "summary": {}, "periods": []
    })
    
    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "Run 2" # Newest first because created_at desc

def test_get_run_detail():
    # Create the run
    create_response = client.post("/api/runs", json={
        "name": "Detail Run",
        "config": {"test": True},
        "summary": {"res": "ok"},
        "periods": [{"period": 0, "val": 1}]
    })
    run_id = create_response.json()["id"]
    
    # Fetch details
    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Detail Run"
    assert len(data["periods"]) == 1
    assert data["periods"][0]["val"] == 1

def test_delete_run():
    # Create the run
    create_response = client.post("/api/runs", json={
        "name": "Delete Me", "config": {}, "summary": {}, "periods": []
    })
    run_id = create_response.json()["id"]
    
    # Check it exists
    assert len(client.get("/api/runs").json()) == 1
    
    # Delete it
    delete_response = client.delete(f"/api/runs/{run_id}")
    assert delete_response.status_code == 200
    
    # Check it's gone
    assert len(client.get("/api/runs").json()) == 0
    
    # Fetching deleted should 404
    assert client.get(f"/api/runs/{run_id}").status_code == 404

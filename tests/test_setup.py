import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def test_api():
    client = TestClient(app)
    yield client  # testing happens here
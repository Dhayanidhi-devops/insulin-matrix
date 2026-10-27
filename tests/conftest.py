import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("INSULIN_MATRIX_HOME", str(tmp_path))
    return tmp_path

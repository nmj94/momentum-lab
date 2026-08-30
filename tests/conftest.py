"""Never write test research history into the developer's real registry."""

import pytest


@pytest.fixture(autouse=True)
def isolated_observation_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MOMENTUM_LAB_REGISTRY_PATH", str(tmp_path / "audit" / "registry.sqlite3"))

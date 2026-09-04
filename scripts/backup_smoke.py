"""Installed-wheel recovery checks, also reused by actual research lifecycle smokes."""

import json
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import momentum_lab
from momentum_lab import StudyRegistry, inspect_run
from momentum_lab.run_control import RunSession


def verify_recovery(output, destination, registry_path):
    """Exercise real installed CLI commands against an already completed workflow."""
    archive = destination / "run.mlbackup.zip"
    restored = destination / "recovered"
    destination.mkdir()

    def invoke(*args):
        result = subprocess.run(
            [sys.executable, "-m", "momentum_lab", "backup", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "DO_NOT_PRINT" not in result.stdout and "test_metrics" not in result.stdout
        return json.loads(result.stdout)

    created = invoke("create", output, "--output", archive, "--acknowledge-sensitive")
    verified = invoke("inspect", archive, "--expected-sha256", created["archive_sha256"])
    assert verified["integrity"] == "verified"
    result = invoke(
        "restore",
        archive,
        "--output",
        restored,
        "--expected-sha256",
        created["archive_sha256"],
        "--acknowledge-sensitive",
    )
    assert result["status"] == "restored_inactive" and not result["active"]
    original = StudyRegistry(registry_path, create=False)
    recovered_path = restored / "payload/registry/research.sqlite3"
    recovered = StudyRegistry(recovered_path, create=False)
    assert recovered.registry_id == original.registry_id
    with closing(sqlite3.connect(original.path)) as before, closing(sqlite3.connect(recovered.path)) as after:
        assert list(before.iterdump()) == list(after.iterdump())
    assert (restored / "payload/run/run_config.json").read_bytes() == (output / "run_config.json").read_bytes()
    assert inspect_run(restored / "payload/run")["status"] == "untracked"
    assert inspect_run(output, verify=True)["integrity"] == "verified"
    assert not list(restored.rglob("owner.lock"))


def main():
    assert "site-packages" in Path(momentum_lab.__file__).resolve().parts, "smoke must use the installed wheel"
    with tempfile.TemporaryDirectory(prefix="momentum-backup-smoke-") as folder:
        root = Path(folder)
        registry = StudyRegistry(root / "registry.sqlite3")
        target = root / "runs/example"
        target.mkdir(parents=True)
        with RunSession(target, "search") as run:
            run.start()
            (target / "run_config.json").write_text(
                json.dumps({"registry_path": str(registry.path), "registry_id": registry.registry_id}), encoding="utf-8"
            )
            (target / "summary.json").write_text('{"secret_score":"DO_NOT_PRINT"}', encoding="utf-8")
            run.complete(["run_config.json", "summary.json"])
        verify_recovery(target, root / "backup", registry.path)
    print(
        "Backup smoke: passed (installed CLI, private full-registry snapshot, verified inactive restore, no overwrite of live history)"
    )


if __name__ == "__main__":
    main()

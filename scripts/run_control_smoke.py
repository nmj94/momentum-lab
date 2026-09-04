"""Installed core-wheel checks for score-free run receipts and real process death."""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Timer

import momentum_lab
from momentum_lab import RunBusyError, inspect_run
from momentum_lab.run_control import RunSession


def main():
    assert "site-packages" in Path(momentum_lab.__file__).resolve().parts, "smoke must use the installed wheel"
    assert importlib.util.find_spec("sklearn") is None, "run control must not require optional ML"
    with tempfile.TemporaryDirectory(prefix="momentum-run-control-smoke-") as folder:
        root = Path(folder)
        output = root / "runs" / "example"

        def invoke(*args, code=0):
            result = subprocess.run(
                [sys.executable, "-m", "momentum_lab", "runs", *map(str, args)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == code, result.stdout + result.stderr
            assert "PRIVATE_SCORE" not in result.stdout
            return json.loads(result.stdout)

        assert invoke("status", output, code=1)["status"] == "not_found"
        assert not list(root.iterdir())
        with RunSession(output, "search") as session:
            session.start()
            assert invoke("status", output, "--verify")["status"] == "running"
            output.mkdir(parents=True)
            (output / "summary.json").write_text('{"PRIVATE_SCORE":1}', encoding="utf-8")
            session.complete(["summary.json"])
        receipt = invoke("history", output, "--verify")
        assert receipt["status"] == "completed" and receipt["integrity"] == "verified"
        assert len(receipt["history"]) == 1
        (output / "summary.json").write_text("changed", encoding="utf-8")
        assert invoke("status", output, "--verify", code=1)["changed_artifacts"] == ["summary.json"]

        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                (
                    "import sys; from momentum_lab.run_control import RunSession\n"
                    "with RunSession(sys.argv[1], 'search', mode='resume') as session:\n"
                    "    session.start()\n"
                    "    print('READY', flush=True)\n"
                    "    sys.stdin.read()\n"
                ),
                str(output),
            ],
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        watchdog = Timer(30, child.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            assert child.stdout.readline().strip() == "READY", "Child failed to acquire ownership"
            try:
                with RunSession(output, "search"):
                    raise AssertionError("Concurrent writer was admitted")
            except RunBusyError:
                pass
        finally:
            watchdog.cancel()
            watchdog.join()
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
        interrupted = invoke("history", output)
        assert interrupted["status"] == "interrupted"
        assert interrupted["attempt"]["finished_at"] is None
        with RunSession(output, "search", mode="resume") as session:
            session.start()
            session.complete(["summary.json"])
        history = inspect_run(output, verify=True)
        assert history["integrity"] == "verified" and len(history["history"]) == 3
        assert history["history"][1]["status"] == "interrupted"
        assert history["history"][1]["error_type"] == "UncleanExit"
        assert history["history"][1]["finished_at"] is None
        assert history["history"][1]["detected_at"] is not None
    print(
        "Run-control smoke: passed (core wheel, CLI, score-free receipts, contention, killed owner, recovery history)"
    )


if __name__ == "__main__":
    main()

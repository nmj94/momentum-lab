"""Atomic single-file publication with independent, exclusively created staging files."""

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_text_output(path, *, encoding="utf-8", newline=None):
    """Keep the prior file on failure; concurrent writers never share a temp file.

    The staging file lives beside its destination so replace is atomic. This is
    a single-file guarantee, not a transaction spanning an entire research run.
    """
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline=newline,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

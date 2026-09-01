from __future__ import annotations

from pathlib import Path


def get_next_run_dir(
    base_dir: str | Path,
    prefix: str = "run",
) -> Path:
    """
    Create the next numbered run directory.

    Example
    -------
    run_001
    run_002
    run_003
    """

    base_dir = Path(base_dir)

    base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = []

    for path in base_dir.iterdir():
        if path.is_dir() and path.name.startswith(f"{prefix}_"):
            try:
                idx = int(path.name.split("_")[1])

                existing.append(idx)

            except Exception:
                pass

    next_idx = 1 if len(existing) == 0 else max(existing) + 1

    run_name = f"{prefix}_{next_idx:03d}"

    run_dir = base_dir / run_name

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_dir

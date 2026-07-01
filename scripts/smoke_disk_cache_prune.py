from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types


def install_fake_modules(output_dir: Path) -> None:
    torch = types.ModuleType("torch")

    class Tensor:
        pass

    torch.Tensor = Tensor
    sys.modules["torch"] = torch

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(output_dir)
    sys.modules["folder_paths"] = folder_paths


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_slot(root: Path, name: str, size: int, age_seconds: float) -> Path:
    slot = root / "SCAIL2ScheduledLongVideo" / name
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "cache.pt").write_bytes(b"x" * int(size))
    (slot / "meta.json").write_text("{}", encoding="utf-8")
    mtime = time.time() - float(age_seconds)
    os.utime(slot / "cache.pt", (mtime, mtime))
    os.utime(slot / "meta.json", (mtime, mtime))
    return slot


def slot_exists(slot: Path) -> bool:
    return (slot / "cache.pt").exists()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    with tempfile.TemporaryDirectory(prefix="scail_cache_prune_") as tmp:
        output_dir = Path(tmp) / "output"
        install_fake_modules(output_dir)
        import nodes

        cache_root = output_dir / "scail2_cache" / "long_video"

        os.environ["SCAIL2_DISK_CACHE_MAX_ENTRIES"] = "2"
        os.environ["SCAIL2_DISK_CACHE_MAX_BYTES"] = "0"
        os.environ["SCAIL2_DISK_CACHE_MAX_AGE_DAYS"] = "0"
        old_1 = make_slot(cache_root, "old_1", 10, 40)
        old_2 = make_slot(cache_root, "old_2", 10, 30)
        new_1 = make_slot(cache_root, "new_1", 10, 20)
        new_2 = make_slot(cache_root, "new_2", 10, 10)
        summary = nodes._prune_scail2_disk_cache()
        assert_true(summary["deleted"] == 2, f"entry prune deleted {summary['deleted']} slots")
        assert_true(not slot_exists(old_1), "oldest slot should be pruned")
        assert_true(not slot_exists(old_2), "second-oldest slot should be pruned")
        assert_true(slot_exists(new_1), "newer slot should remain")
        assert_true(slot_exists(new_2), "newest slot should remain")

        shutil.rmtree(cache_root)
        os.environ["SCAIL2_DISK_CACHE_MAX_ENTRIES"] = "0"
        os.environ["SCAIL2_DISK_CACHE_MAX_BYTES"] = "250"
        os.environ["SCAIL2_DISK_CACHE_MAX_AGE_DAYS"] = "0"
        protected = make_slot(cache_root, "protected_old", 100, 30)
        middle = make_slot(cache_root, "middle", 100, 20)
        newest = make_slot(cache_root, "newest", 100, 10)
        summary = nodes._prune_scail2_disk_cache(str(protected / "cache.pt"))
        assert_true(summary["deleted"] == 1, f"byte prune deleted {summary['deleted']} slots")
        assert_true(slot_exists(protected), "protected current cache should remain")
        assert_true(not slot_exists(middle), "old unprotected slot should be pruned to fit byte cap")
        assert_true(slot_exists(newest), "newest unprotected slot should remain")

        shutil.rmtree(cache_root)
        os.environ["SCAIL2_DISK_CACHE_MAX_ENTRIES"] = "0"
        os.environ["SCAIL2_DISK_CACHE_MAX_BYTES"] = "0"
        os.environ["SCAIL2_DISK_CACHE_MAX_AGE_DAYS"] = "1"
        expired = make_slot(cache_root, "expired", 10, 2 * 86400)
        fresh = make_slot(cache_root, "fresh", 10, 60)
        summary = nodes._prune_scail2_disk_cache()
        assert_true(summary["deleted"] == 1, f"age prune deleted {summary['deleted']} slots")
        assert_true(not slot_exists(expired), "expired slot should be pruned")
        assert_true(slot_exists(fresh), "fresh slot should remain")

    print("smoke_disk_cache_prune: ok")


if __name__ == "__main__":
    main()

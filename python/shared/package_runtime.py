import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False


SHARED_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = SHARED_DIR.parent
PACKAGE_ROOT = PYTHON_ROOT.parent

DATA_DIR = PACKAGE_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
MANIFESTS_DIR = DATA_DIR / "manifests"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
RUNS_DIR = PACKAGE_ROOT / "runs"
LOGS_DIR = RUNS_DIR / "logs"

PACKAGE_IMPORT_DIRS = [
    PYTHON_ROOT,
    PYTHON_ROOT / "shared",
    PYTHON_ROOT / "agent_support",
    PYTHON_ROOT / "care_common",
    PYTHON_ROOT / "care_workflow",
    PYTHON_ROOT / "eval",
]


def bootstrap_import_paths() -> None:
    for p in PACKAGE_IMPORT_DIRS:
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)


def load_package_env() -> Path:
    env_path = PACKAGE_ROOT / ".env"
    load_dotenv(env_path, override=False)
    return env_path


def ensure_runtime_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


class _TeeTextIO:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextmanager
def tee_console_to_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "a", encoding="utf-8") as f:
        tee_out = _TeeTextIO(original_stdout, f)
        tee_err = _TeeTextIO(original_stderr, f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            yield log_path


def resolve_package_sample_path(sample_path: str | os.PathLike[str] | None = None) -> Path | None:
    if not sample_path:
        return None
    path = Path(sample_path)
    if path.is_absolute():
        return path.resolve()
    return (PACKAGE_ROOT / path).resolve()


def _quick_sample_parent_feature_store(sample_path: Path) -> Path | None:
    name = sample_path.name
    if "_quick_n" not in name:
        return None
    candidate_name = name.replace("_eval_pair_quick_n200_", "_eval_pair_n1000_").replace(".csv", "__full_feature_package.csv")
    candidate = FEATURE_STORE_DIR / candidate_name
    if candidate.exists():
        return candidate
    return None


def resolve_feature_store_for_sample(sample_path: Path | str | None) -> Path | None:
    env_path = os.getenv("FEATURE_STORE_CSV", "").strip()
    if env_path:
        return resolve_package_sample_path(env_path)

    if sample_path is None:
        return None

    path = Path(sample_path).resolve()
    direct = FEATURE_STORE_DIR / f"{path.stem}__full_feature_package.csv"
    if direct.exists():
        return direct

    parent = _quick_sample_parent_feature_store(path)
    if parent is not None:
        return parent

    return None


def resolve_duckdb_path() -> Path:
    env_path = os.getenv("MIMIC_DB_PATH", "").strip()
    if env_path:
        return resolve_package_sample_path(env_path)
    return (DATA_DIR / "mimiciv.duckdb").resolve()


def resolve_feature_source() -> str:
    source = os.getenv("FEATURE_SOURCE", "auto").strip().lower()
    if source not in {"auto", "duckdb", "locked_csv"}:
        raise ValueError("FEATURE_SOURCE must be one of: auto, duckdb, locked_csv")
    return source


def source_requires_duckdb() -> bool:
    source = resolve_feature_source()
    return source in {"duckdb", "auto"}


def resolve_prompt_dir(subdir: str | None = None) -> Path:
    if subdir:
        return (PROMPTS_DIR / subdir).resolve()
    return PROMPTS_DIR.resolve()

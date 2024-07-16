from pathlib import Path


def project_root() -> Path:
    Path(__file__).parent.parent.mkdir(exist_ok=True, parents=True)
    return Path(__file__).parent.parent


def readme_file() -> Path:
    return project_root() / "README.md"

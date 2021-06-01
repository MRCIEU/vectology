from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Dict


def find_project_root(anchor_file: str = "setup.py") -> Path:
    cwd = Path.cwd()
    test_dir = cwd
    prev_dir = None
    while prev_dir != test_dir:
        if (test_dir / anchor_file).exists():
            return test_dir
        prev_dir = test_dir
        test_dir = test_dir.parent
    return cwd


def check_dependent_files(dependency_spec: Dict) -> None:
    input_exists = {path: path.exists() for path in dependency_spec["input"]}
    output_exists = {path: path.exists() for path in dependency_spec["output"]}
    overall = {
        "input": input_exists,
        "output": output_exists,
    }
    pprint(overall)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

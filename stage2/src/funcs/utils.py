import datetime
import io  # for printing df info
from pathlib import Path

import pandas as pd
from icecream import ic
from loguru import logger


def ic_timestamp() -> str:
    res = "{time} |> ".format(time=str(datetime.datetime.now()))
    return res


ic.configureOutput(prefix=ic_timestamp)


def find_project_root(anchor_file: str = "environment.yml") -> Path:
    cwd = Path.cwd()
    test_dir = cwd
    prev_dir = None
    while prev_dir != test_dir:
        if (test_dir / anchor_file).exists():
            return test_dir
        prev_dir = test_dir
        test_dir = test_dir.parent
    return cwd


def find_data_root() -> Path:
    proj_root = find_project_root()
    path = proj_root / "data"
    if not path.exists():
        logger.info(f"Path {path} does not exists")
    return path


def find_analysis_artifacts_dir() -> Path:
    data_root = find_data_root()
    path = data_root / "analysis-artifacts"
    if not path.exists():
        logger.info(f"Path {path} does not exists")
    return path


def df_info(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    return s


def df_shape(df: pd.DataFrame) -> pd.DataFrame:
    print(df.shape)
    return df

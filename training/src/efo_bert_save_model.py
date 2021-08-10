import argparse
from pathlib import Path

from loguru import logger

from funcs.efo.efo_bert_model import Model
from funcs.utils import find_project_root

ROOT = find_project_root()
DEFAULT_EXPORT_PATH = ROOT / "models" / "efo_bert" / "model"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--export-path",
        default=DEFAULT_EXPORT_PATH,
        type=Path,
        help="default: %(default)s",
    )
    args = parser.parse_args()
    logger.info(args)
    model_path = args.model_path
    export_path = args.export_path
    assert model_path.exists()

    logger.info(f"Load model from {model_path}")
    model = Model().load_from_checkpoint(str(model_path)).bert_model

    export_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(export_path))
    logger.info(f"Save model to {export_path}")


if __name__ == "__main__":
    main()

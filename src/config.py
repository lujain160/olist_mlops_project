import yaml
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path=None):
    try:
        if config_path is None:
            root_dir = Path(__file__).resolve().parent.parent
            config_path = root_dir / "config" / "config.yaml"
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileExistsError(f"Configuration file not found at {config_path}")
        logger.info(f"Loading configuration from {config_path}...")
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(
            f"Error occurred while loading configuration: {str(e)}", exc_info=True
        )
        raise e

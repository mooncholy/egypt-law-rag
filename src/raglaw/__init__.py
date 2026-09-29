from .logging_conf import get_logger

logger = get_logger(__name__)


def log_package_banner() -> None:
    """
    Emit one line proving the package imports and its logging is configured.

    Left over from the initial packaging step. Nothing in the service calls it,
    and it can go once the packaging check is no longer useful.
    """
    logger.info("Hello from mlops-practitioner!")

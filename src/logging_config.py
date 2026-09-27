import logging

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """configure console logging for the training pipeline"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True

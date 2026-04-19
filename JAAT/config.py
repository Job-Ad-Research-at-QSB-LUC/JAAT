import logging

logger = logging.getLogger("JAAT")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

GLOBAL_SETTINGS = {
    "show_progress": True
}

MODEL_CACHE = {}
import logging
import os

# Create the logger
logger = logging.getLogger("main_logger")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    # Create a file handler to log to a file
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/training.log")
    file_handler.setLevel(logging.DEBUG)

    # Define log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)

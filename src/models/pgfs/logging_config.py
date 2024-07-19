import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Define a flag to track whether the logging has been set up
is_logging_setup = False


def setup_logging(log_dir="/rds/user/gtj21/hpc-work/designing-new-molecules/logs"):
    global is_logging_setup
    if is_logging_setup:
        return
    is_logging_setup = True

    os.makedirs(log_dir, exist_ok=True)

    # Determine the log file name
    log_file_name = f"moldesign_traintd3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    full_log_path = os.path.join(log_dir, log_file_name)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set the root logger level

    # Clear existing handlers
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[-1])

    # Create rotating file handler
    file_handler = RotatingFileHandler(
        full_log_path, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Create stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter(log_format))

    # Create stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(logging.Formatter(log_format))

    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)

    # Log that the logging has been set up
    root_logger.info(f"Logging setup complete. Log file: {log_file_name}")
    root_logger.info(f" New run started at {datetime.now()}")


# Ensure this script sets up logging when it runs
if __name__ == "__main__":
    setup_logging()

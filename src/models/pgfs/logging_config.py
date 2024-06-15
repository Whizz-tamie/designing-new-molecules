import logging
import os
import sys
from datetime import datetime

log_file_name = None

def setup_logging():
    log_dir = "/rds/user/gtj21/hpc-work/designing-new-molecules/logs"
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if 'LOG_FILE_NAME' not in os.environ:
        os.makedirs(log_dir, exist_ok=True)
        log_file_name = f"pgfs_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        os.environ['LOG_FILE_NAME'] = log_file_name
    else:
        log_file_name = os.environ['LOG_FILE_NAME'] 
    
    # Clear all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)

    file_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)

    # Ensure the log file is created by writing an initial message
    root_logger.info(f"Logging setup complete. Log file: {log_file_name}")
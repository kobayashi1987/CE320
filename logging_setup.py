import os
from datetime import datetime
import logging


def setup_logging(output_path):
    """
    Configures logging to both file and console.
    """
    log_dir = os.path.join(output_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console_handler)


logger = logging.getLogger(__name__)

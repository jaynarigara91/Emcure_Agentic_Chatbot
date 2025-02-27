import os
from datetime import datetime
import logging

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

log_file_path = os.path.join(log_dir,log_file)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - (Line : %(lineno)d))'
)
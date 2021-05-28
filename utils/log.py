import logging
import sys
from datetime import datetime


# initiate the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
time_flag= datetime.utcnow().strftime('%Y-%m-%d-%H:%M') 
logging.basicConfig(level=logging.INFO, format=FORMAT, filename="log/training_"+time_flag)
logger = logging.getLogger(__name__)
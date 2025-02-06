from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import sys

class ColoredFormatter(logging.Formatter):
   green = "\x1b[32m"
   reset = "\x1b[0m"
   
   def __init__(self):
       super().__init__(
           fmt='%(colored_level)s: -- %(name)s -- %(message)s',
           datefmt='%H:%M:%S'
       )

   def format(self, record):
       if record.levelname == "INFO":
           record.colored_level = f"{self.green}INFO{self.reset}"
       else:
           record.colored_level = record.levelname
       if record.msg:
           record.msg = record.msg.replace('\n', ' ')
       return super().format(record)

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(ColoredFormatter())
            logger.addHandler(file_handler)

    return logger

class BaseBench(ABC):
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(f"{self.__class__.__name__} Initialized!")

    @abstractmethod
    def run_bench(self, agent_url: str, task_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_all_tasks(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass

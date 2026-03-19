from dataclasses import dataclass
from typing import Union, Optional
# import config
from . import config

@dataclass
class Config():
    data_save_path: Optional[str] = config.data_save_path
    data_load_path: Optional[str] = config.data_load_path
    


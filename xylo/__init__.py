from dataclasses import dataclass
from typing import Union, Optional
# import config
from . import config
from . import features
from . import datastructure
from . import training

@dataclass
class Config():
    okeon_bucket: Optional[str] = config.okeon_bucket
    


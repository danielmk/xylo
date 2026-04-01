import platform
import pathlib

if platform.system() == 'Windows':
    okeon_bucket = pathlib.Path("Y:\danielmk\okeon")
elif platform.system() == 'Linux':
    okeon_bucket = pathlib.Path("/bucket/FukaiU/danielmk/okeon/")


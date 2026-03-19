import platform
import os

if platform.system() == 'Windows':
    data_save_path = os.path.join(os.path.dirname(__file__), 'data')
    data_load_path = os.path.join(os.path.dirname(__file__), 'data')
elif platform.system() == 'Linux':
    data_save_path = r'/'
    data_load_path = r'/'


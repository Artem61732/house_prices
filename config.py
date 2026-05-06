from omegaconf import OmegaConf
import os

# Получаем абсолютный путь к файлу config.yaml (чтобы не было проблем с путями)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

# Загружаем конфигурацию
cfg = OmegaConf.load(CONFIG_PATH)
"""
项目配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 通义千问API配置
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
QWEN_MODEL = os.environ.get('QWEN_MODEL', 'qwen-plus')

# 默认路径
DEFAULT_DICT_PATH = PROJECT_ROOT / 'gloss_dict.npy'
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / 'slr_mindspore.ckpt'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'output_dir'

# API配置
API_TIMEOUT = 30  # 秒
API_MAX_RETRIES = 3

# LLM生成参数
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500

# 视频预处理参数
VIDEO_CROP_SIZE = 224
VIDEO_LEFT_PAD = 6

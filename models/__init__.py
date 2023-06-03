import importlib
import os.path as osp

from utils import get_root_logger, scandir
from utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
# 通过扫描 model_folder 目录下以 _model.py 结尾的文件，将这些文件的基本名称（去除扩展名）存储在 model_filenames 列表中
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
# 通过循环遍历 model_filenames 列表中的文件名，使用 importlib.import_module 函数动态导入 models 包下对应文件的模块，
# 并将这些模块存储在 _model_modules 列表中
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """
    Build model from options
    Args:
        opt (dict): Configuration dict. It must contain:
            type (str): Model type.

    returns:
        model (BaseModel): model built by opt.
    """
    # 根据传入的配置选项 opt 中的模型类型，从 MODEL_REGISTRY 中获取相应的模型构造函数，并使用 opt 构建模型对象 model
    model = MODEL_REGISTRY.get(opt['type'])(opt)
    logger = get_root_logger()
    # 使用日志记录器 logger 记录模型的创建信息，包括模型类的名称
    logger.info(f'Model [{model.__class__.__name__}] is created.')

    return model

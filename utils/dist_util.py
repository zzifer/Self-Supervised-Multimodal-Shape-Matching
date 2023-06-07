# distributed training utils support for DistributedDataParallel (ddp)
import functools
import os
import re

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# # 初始化分布式训练环境
def init_dist(backend='nccl', port=29500):
    """Initialize slurm distributed training environment.
    Args:
        backend (str, optional): Backend of torch.distributed. Default 'nccl'.
        port (int, optional): the port number for tcp/ip communication. Default 29500.
    """
    # get_start_method(allow_none=True)获取当前的多进程启动方法，如果未设置则设置为‘spawn’
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    _init_dist_slurm(backend, port)

    
# 用于在Slurm分布式训练环境下初始化分布式设置
def _init_dist_slurm(backend, port):
    # 1. get environment info
    # RANK：当前进程的序号，用于进程间通讯，rank = 0 的主机为 master 节点
    rank = int(os.environ['SLURM_PROCID'])
    # 获取总进程数
    world_size = int(os.environ['SLURM_NTASKS'])
    # LOCAL_RANK：当前进程对应的GPU号
    local_rank = int(os.environ['SLURM_LOCALID'])
    # 获取节点列表，通过环境变量SLURM_NODELIST获取
    node_list = str(os.environ['SLURM_NODELIST'])

    # 2. specify ip address
    # 使用正则表达式从节点列表中提取数字部分
    node_parts = re.findall('[0-9]+', node_list)
    # 根据提取的数字部分构造主机IP地址
    host_ip = '{}.{}.{}.{}'.format(node_parts[1], node_parts[2], node_parts[3], node_parts[4])

    # 3. specify port number
    # 将端口号转换为字符串形式
    port = str(port)

    # 4. initialize tcp/ip communication
    # 根据主机IP地址和端口号构造初始化方法
    init_method = 'tcp://{}:{}'.format(host_ip, port)
    try:
        # 使用指定的后端、初始化方法、总进程数和当前进程的rank初始化进程组
        dist.init_process_group(backend, init_method=init_method, world_size=world_size, rank=rank)
    except:
        raise ValueError(f'Initialize DDP failed. The port {port} is already used. Please assign a different port.')

    # 5. specify current device
    # 设置当前设备为指定的本地rank对应的GPU设备
    torch.cuda.set_device(local_rank)


# 用于限制函数只在主节点（rank=0）上执行
def master_only(func):
    """
    Function only executes in the master rank (rank = 0).

    Args:
        func (Callable): callable function
    """

    # 装饰器函数的语法，用于将装饰器应用到下面的函数
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # get rank
        rank, _ = get_dist_info()

        # execute only when rank = 0
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


# 用于获取分布式信息
def get_dist_info():
    """
    Get distribution information.

    Returns:
        rank (int): the rank number of current process group.
        world_size (int): the total number of the processes.
    """
    # 检查分布式训练模块是否可用并是否已初始化
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    # 如果已经初始化，则获取当前进程的rank和总进程数
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size

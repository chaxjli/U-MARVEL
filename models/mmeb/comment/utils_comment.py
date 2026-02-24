import torch
from src.logging import get_logger
logger = get_logger(__name__)

def print_rank(message):
    """
    在分布式训练环境中打印带有进程排名前缀的消息，或在非分布式环境中直接打印消息。
    
    参数:
        message (str): 需要打印的消息内容
    
    行为:
        - 当分布式环境已初始化时，在消息前添加"rankX: "前缀（X为当前进程的排名）
        - 当分布式环境未初始化时，直接打印原始消息
    """
    if torch.distributed.is_initialized():
        # 获取当前进程的排名并添加到消息前缀
        logger.info(f'rank{torch.distributed.get_rank()}: ' + message)
    else:
        # 非分布式环境直接打印消息
        logger.info(message)


def print_master(message):
    """
    确保消息仅在分布式环境的主进程（rank 0）中打印，或在非分布式环境中直接打印。
    
    参数:
        message (str): 需要打印的消息内容
    
    行为:
        - 当分布式环境已初始化时:
            - 如果当前进程是主进程（rank 0），则打印消息
            - 如果当前进程不是主进程，则忽略该消息
        - 当分布式环境未初始化时，直接打印原始消息
    """
    if torch.distributed.is_initialized():
        # 仅在主进程(rank 0)中打印消息
        if torch.distributed.get_rank() == 0:
            logger.info(message)
    else:
        # 非分布式环境直接打印消息
        logger.info(message)
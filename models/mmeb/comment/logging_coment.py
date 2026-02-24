# https://github.com/huggingface/datasets/blob/main/src/datasets/utils/logging.py
# 版权声明
# Copyright 2020 Optuna, HuggingFace
#
# 使用Apache License 2.0许可证
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则按"原样"分发软件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""日志记录工具。"""
# 模块文档字符串，说明这是一个日志记录工具模块

import logging
# 导入Python标准库的logging模块

# 配置基础日志格式
# 格式包括：时间 - 日志级别 - 日志名称 - 消息
# 时间格式：月/日/年 时:分:秒
# 默认日志级别为WARN
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.WARN
)

import os
# 导入操作系统相关功能模块

# 从logging模块导入各种日志级别常量
# NOQA注释表示这些导入是必要的，不应被linter标记为未使用
from logging import (
    CRITICAL,  # NOQA - 严重错误级别
    DEBUG,     # NOQA - 调试级别
    ERROR,     # NOQA - 错误级别
    FATAL,     # NOQA - 致命错误级别(等同于CRITICAL)
    INFO,      # NOQA - 信息级别
    NOTSET,    # NOQA - 未设置级别
    WARN,      # NOQA - 警告级别(等同于WARNING)
    WARNING,   # NOQA - 警告级别
)
from typing import Optional
# 导入Optional类型提示，用于类型注解

from tqdm import auto as tqdm_lib
# 导入tqdm库的auto模块，用于进度条显示

# 定义日志级别名称到logging模块常量的映射字典
log_levels = {
    "debug": logging.DEBUG,      # 调试级别
    "info": logging.INFO,        # 信息级别
    "warning": logging.WARNING,  # 警告级别
    "error": logging.ERROR,      # 错误级别
    "critical": logging.CRITICAL, # 严重错误级别
}

# 默认日志级别为INFO
_default_log_level = logging.INFO


def _get_default_logging_level():
    """
    如果设置了DATASETS_VERBOSITY环境变量且值为有效选项之一，则返回对应的日志级别。
    如果未设置或值无效，则返回默认日志级别``_default_log_level``。
    
    返回:
        int: 对应的日志级别常量
    """
    # 从环境变量获取日志级别设置
    env_level_str = os.getenv("DATASETS_VERBOSITY", None)
    if env_level_str:
        # 检查环境变量值是否在预定义的日志级别中
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            # 如果值无效，记录警告
            logging.getLogger().warning(
                f"Unknown option DATASETS_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    # 返回默认日志级别
    return _default_log_level


def _get_library_name() -> str:
    """
    获取库的名称（从模块的__name__中提取第一个部分）。
    
    返回:
        str: 库的名称
    """
    return __name__.split(".")[0]


def _get_root_logger() -> logging.Logger:
    """
    获取库的根日志记录器。
    
    返回:
        logging.Logger: 根日志记录器实例
    """
    # 返回以库名称命名的日志记录器
    # 注释掉的代码是原始实现，现在直接返回根日志记录器
    # return logging.getLogger(_get_library_name())
    return logging.getLogger()


def _configure_root_logger() -> None:
    """
    配置根日志记录器，应用默认设置。
    """
    # 获取根日志记录器
    root_logger = _get_root_logger()
    # 注释掉的代码是添加流处理器的原始实现
    # root_logger.addHandler(logging.StreamHandler())
    # 设置日志级别为从环境变量或默认值获取的级别
    root_logger.setLevel(_get_default_logging_level())


def _reset_root_logger() -> None:
    """
    重置根日志记录器，将日志级别设置为NOTSET。
    """
    root_logger = _get_root_logger()
    root_logger.setLevel(logging.NOTSET)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取指定名称的日志记录器。
    此函数可在数据集脚本中使用。
    
    参数:
        name (str, optional): 日志记录器名称。如果为None，则使用库名称。
    
    返回:
        logging.Logger: 日志记录器实例
    """
    if name is None:
        name = _get_library_name()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """
    获取HuggingFace datasets库根日志记录器的当前日志级别。
    
    返回:
        int: 当前日志级别，如datasets.logging.DEBUG或datasets.logging.INFO。
    
    <Tip>
        HuggingFace datasets库有以下日志级别:
        - datasets.logging.CRITICAL, datasets.logging.FATAL
        - datasets.logging.ERROR
        - datasets.logging.WARNING, datasets.logging.WARN
        - datasets.logging.INFO
        - datasets.logging.DEBUG
    </Tip>
    """
    return _get_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """
    设置HuggingFace Datasets库根日志记录器的日志级别。
    
    参数:
        verbosity (int): 日志级别，如datasets.logging.DEBUG或datasets.logging.INFO。
    """
    _get_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """
    将HuggingFace datasets库根日志记录器的级别设置为INFO。
    这将显示大多数日志信息和tqdm进度条。
    
    等同于datasets.logging.set_verbosity(datasets.logging.INFO)的快捷方式。
    """
    return set_verbosity(INFO)


def set_verbosity_warning():
    """
    将HuggingFace datasets库根日志记录器的级别设置为WARNING。
    这将仅显示警告和错误日志信息以及tqdm进度条。
    
    等同于datasets.logging.set_verbosity(datasets.logging.WARNING)的快捷方式。
    """
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """
    将HuggingFace datasets库根日志记录器的级别设置为DEBUG。
    这将显示所有日志信息和tqdm进度条。
    
    等同于datasets.logging.set_verbosity(datasets.logging.DEBUG)的快捷方式。
    """
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """
    将HuggingFace datasets库根日志记录器的级别设置为ERROR。
    这将仅显示错误日志信息和tqdm进度条。
    
    等同于datasets.logging.set_verbosity(datasets.logging.ERROR)的快捷方式。
    """
    return set_verbosity(ERROR)


def disable_propagation() -> None:
    """
    禁用库日志输出的传播。
    注意日志传播默认是禁用的。
    """
    _get_root_logger().propagate = False


def enable_propagation() -> None:
    """
    启用库日志输出的传播。
    如果根日志记录器已经配置，请禁用HuggingFace datasets库的默认处理器以避免重复日志记录。
    """
    _get_root_logger().propagate = True


# 在模块级别配置库的根日志记录器（类似单例模式）
_configure_root_logger()


class EmptyTqdm:
    """
    一个不执行任何操作的虚拟tqdm类。
    用于在禁用进度条时提供兼容接口。
    """

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        # 保存传入的迭代器（如果有）
        self._iterator = args[0] if args else None

    def __iter__(self):
        # 返回原始迭代器
        return iter(self._iterator)

    def __getattr__(self, _):
        """
        返回一个空函数，用于拦截所有未定义的方法调用。
        """
        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return
        return empty_fn

    def __enter__(self):
        # 上下文管理器入口
        return self

    def __exit__(self, type_, value, traceback):
        # 上下文管理器出口
        return


# 全局变量，控制tqdm进度条是否激活
_tqdm_active = True


class _tqdm_cls:
    """
    自定义tqdm类，根据_tqdm_active状态决定返回真实tqdm还是EmptyTqdm。
    """
    def __call__(self, *args, disable=False, **kwargs):
        # 如果全局激活且未显式禁用，则返回真实tqdm
        if _tqdm_active and not disable:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            # 否则返回空tqdm
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        # 设置锁，如果tqdm激活则设置真实锁，否则不执行操作
        self._lock = None
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        # 获取锁，如果tqdm激活则返回真实锁
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()

    def __delattr__(self, attr):
        """
        修复https://github.com/huggingface/datasets/issues/6066问题
        特殊处理_lock属性的删除，其他属性正常处理
        """
        try:
            del self.__dict__[attr]
        except KeyError:
            if attr != "_lock":
                raise AttributeError(attr)


# 创建_tqdm_cls实例作为模块的tqdm接口
tqdm = _tqdm_cls()


def is_progress_bar_enabled() -> bool:
    """
    返回一个布尔值，指示是否启用了tqdm进度条。
    
    返回:
        bool: 如果进度条启用则为True，否则为False
    """
    global _tqdm_active
    return bool(_tqdm_active)


def enable_progress_bar():
    """
    启用tqdm进度条。
    """
    global _tqdm_active
    _tqdm_active = True


def disable_progress_bar():
    """
    禁用tqdm进度条。
    """
    global _tqdm_active
    _tqdm_active = False
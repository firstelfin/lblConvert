#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2025/06/29 11:11:47

import warnings
import cv2
import numpy as np
from typing import Dict, List
from pathlib import Path
from copy import deepcopy
from typing import Union, Optional, Any
from numpy import ndarray
from tqdm import tqdm
from tqdm.std import tqdm as std_tqdm
from loguru import logger
from functools import partial
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
from lblConvert.tools import CPU_KERNEL_NUM
warnings.filterwarnings('ignore')


def colorstr(*args):
    r"""Copy from https://github.com/ultralytics
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.
    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')
    In the second form, 'blue' and 'bold' will be applied by default.
    Args:
        *args (str | Path): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.
    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'
    Returns:
        (str): The args string wrapped with ANSI escape codes for the specified color and style.
    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"
    """

    assert len(args) > 0, "colorstr() takes at least one argument (the string to color)"
    *args, string = args if len(args) > 1 else ("blue", "bold", args[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def load_img(img_path: Union[str, Path]) -> Optional[ndarray]:
    """加载图像文件

    :param Path|str img_path: 图像文件路径
    :return ndarray: 图像数据
    """

    if img_path is None:
        return None
    if not Path(img_path).exists():
        return None
    
    for _ in range(3):
        img = cv2.imread(str(img_path))
        if img is not None:
            return img
    return None


def segmentation_to_polygons(
        annotation: Dict,
        min_area: float = 0.0) -> List[List[List[float]]]:
    """
    将 COCO annotation 中的 segmentation 转换为多边形列表
    
    参数:
        annotation: COCO 格式的 annotation 字典
        min_area: 最小保留面积（过滤太小的多边形）
    
    返回:
        polygons: 多边形列表 [[x1, y1, x2, y2, ...], ...]
    """
    segmentation = annotation['segmentation']
    
    # ========== 情况 1: 已经是多边形格式 ==========
    if isinstance(segmentation, list):
        filtered_polygons = []
        max_area = 0
        for poly in segmentation:
            if len(poly) < 6:
                continue
            poly_arr = np.array(poly).reshape(-1, 2)
            area = cv2.contourArea(poly_arr)
            if min_area > 0 and area < min_area:
                continue
            if area > max_area:
                filtered_polygons.insert(0, poly_arr.tolist())
            else:
                filtered_polygons.append(poly_arr.tolist())
        return filtered_polygons
    
    # ========== 情况 2: RLE 格式 ==========
    elif isinstance(segmentation, Dict):
        # 准备 RLE 数据（处理 Python 3 的 bytes 问题）
        rle = segmentation.copy()
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        
        # 解码为二值掩码
        binary_mask = mask_utils.decode(rle).astype(np.uint8)  # type: ignore[arg-type]
        
        # 使用 OpenCV 提取轮廓. TODO: 多个多边形适配
        # 提取所有轮廓
        contours, hierarchy = cv2.findContours(
            binary_mask, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        hierarchy = hierarchy[0]  # 去掉第一维
        external_contours = []  # 外部轮廓
        hole_contours = []      # 内部孔洞

        for cnt, hier in zip(contours, hierarchy):
            if hier[-1] == -1:
                # 没有父级 = 外部轮廓
                external_contours.append(cnt)
            else:
                # 有父级 = 内部孔洞
                hole_contours.append(cnt)
        contours = external_contours + hole_contours
        
        # 将轮廓转换为 COCO 多边形格式
        polygons = []
        for contour in contours:
            # 过滤太小的轮廓
            if min_area > 0:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
            
            # 轮廓点数为偶数才能构成有效多边形
            if len(contour) < 3:
                continue
            
            # 展平为 [x1, y1, x2, y2, ...] 格式
            polygon = contour.reshape(-1, 2).tolist()
            polygons.append(polygon)
        
        return polygons
    
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")


class TqdmFutureCallback(object):
    """多线程、多进程任务进度条回调函数

    :param int timeout: future 超时时间, 默认20秒
    """

    def __init__(self, timeout: int = 20):
        self.future_error = list()
        self.timeout = timeout
    
    def __call__(
            self, future: Future, bar: std_tqdm, param_args, param_kwargs, 
            results: List[Any], index: int, *args, **kwargs
        ):
        try:
            result = future.result(timeout=self.timeout)
            results[index] = result
        except Exception as e:
            self.future_error.append((param_args, param_kwargs, e, index))
        finally:
            bar.update(1)


class FutureBar(object):
    """多进程、多线程并发执行任务, 并显示进度条. 进度条统一管理类, 异步对象错误收集重试自动化.

    :param int max_workers: 最大并发数, 默认为None, 自动根据CPU核数设置
    :param bool use_process: 是否使用多进程, 默认为False, 即使用多线程
    :param int timeout: 异步任务超时时间, 默认为20秒

    其他参数全部是
    """

    def __init__(
            self, max_workers=None, use_process=False, timeout=20,
            iterable=None, total=None, desc=None, colour="#CD8500",
            leave=True, file=None, ncols=None, mininterval=0.1, 
            maxinterval=10.0, miniters=None, ascii=None, disable=False, 
            unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.3, 
            bar_format=None, initial=0, position=None, postfix=None, 
            unit_divisor=1000, write_bytes=False, lock_args=None, nrows=None, 
            delay=0, gui=False, **kwargs
        ):
        self.max_workers = max_workers if isinstance(max_workers, int) else max(CPU_KERNEL_NUM // 2, 6)
        self.use_process = use_process
        self.bar_callback = TqdmFutureCallback(timeout=timeout)
        new_desc = colorstr("bright_blue", "bold", desc) if isinstance(desc, str) else desc
        self.bar_kwargs = {
            "iterable": iterable, "total": total, "desc": new_desc, "colour": colour,
            "leave": leave, "file": file, "ncols": ncols, "mininterval": mininterval,
            "maxinterval": maxinterval, "miniters": miniters, "ascii": ascii, "disable": disable,
            "unit": unit, "unit_scale": unit_scale, "dynamic_ncols": dynamic_ncols, "smoothing": smoothing,
            "bar_format": bar_format, "initial": initial, "position": position, "postfix": postfix,
            "unit_divisor": unit_divisor, "write_bytes": write_bytes, "lock_args": lock_args, "nrows": nrows,
            "delay": delay, "gui": gui,
        }
        self.bar_kwargs.update(kwargs)
    
    def init_bar(self):
        self.bar = tqdm(**self.bar_kwargs)
        return self.bar

    def get_concurrent_executor(self):
        if self.use_process:
            return ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            return ThreadPoolExecutor(max_workers=self.max_workers)
    
    def retry_failed_tasks(self, exec_func, results):
        if self.bar_callback.future_error:
            logger.warning(f"There are {len(self.bar_callback.future_error)} errors in the concurrent tasks.")
        else:
            return None
        
        logger.info(f"Retrying {len(self.bar_callback.future_error)} tasks...")
        for param_args, param_kwargs, e, idx in self.bar_callback.future_error:
            result = exec_func(*param_args, **param_kwargs)
            results[idx] = result

    def __call__(self, exec_func, params, *args, **kwargs):
        """自定义多进程、多线程执行接口

        :param callable exec_func: 执行函数
        :param iterable params: 参数列表[可迭代对象], 每个元素包含一个参数元组(args, kwargs)
        """
        total = len(list(deepcopy(params))) if "total" not in kwargs else kwargs["total"]
        self.bar_kwargs.update({"total": total})
        self.bar = self.init_bar()
        results: List[Any] = [None] * total
        with self.get_concurrent_executor() as executor:  # TODO: 多进程执行是有问题的, 循环直接退出没有等待执行
            for idx, (param_args, param_kwargs) in enumerate(params):
                future = executor.submit(exec_func, *param_args, **param_kwargs)
                callback = partial(
                    self.bar_callback, bar=self.bar,
                    param_args=param_args, param_kwargs=param_kwargs,
                    results = results, index=idx)
                future.add_done_callback(callback)

        self.bar.close()
        self.retry_failed_tasks(exec_func=exec_func, results=results)
        return results

from __future__ import annotations

import base64
import math
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

# 定义图像缩放因子，用于确保图像的高度和宽度是该因子的倍数
IMAGE_FACTOR = 28
# 定义图像最小像素数，图像缩放后的总像素数不能低于此值
MIN_PIXELS = 4 * 28 * 28
# 定义图像最大像素数，图像缩放后的总像素数不能高于此值，可根据需要调整以减少视觉令牌数量
MAX_PIXELS = 300 * 28 * 28  
# MAX_PIXELS = 5000 * 28 * 28
# 定义图像最大宽高比，超过此比例会引发错误
MAX_RATIO = 200 

# 定义视频最小像素数
VIDEO_MIN_PIXELS = 128 * 28 * 28
# 定义视频最大像素数
VIDEO_MAX_PIXELS = 768 * 28 * 28
# 定义视频总像素数
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
# 定义视频帧缩放因子
FRAME_FACTOR = 2
# 定义视频帧率
FPS = 2.0
# 定义视频最小帧数
FPS_MIN_FRAMES = 4
# 定义视频最大帧数
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """
    返回最接近 'number' 且能被 'factor' 整除的整数。
    :param number: 要处理的整数
    :param factor: 因子
    :return: 最接近 'number' 且能被 'factor' 整除的整数
    """
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """
    返回大于或等于 'number' 且能被 'factor' 整除的最小整数。
    :param number: 要处理的整数
    :param factor: 因子
    :return: 大于或等于 'number' 且能被 'factor' 整除的最小整数
    """
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """
    返回小于或等于 'number' 且能被 'factor' 整除的最大整数。
    :param number: 要处理的整数
    :param factor: 因子
    :return: 小于或等于 'number' 且能被 'factor' 整除的最大整数
    """
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    重新缩放图像，使其满足以下条件：
    1. 高度和宽度都能被 'factor' 整除。
    2. 图像的总像素数在 ['min_pixels', 'max_pixels'] 范围内。
    3. 尽可能保持图像的宽高比。
    :param height: 图像的原始高度
    :param width: 图像的原始宽度
    :param factor: 缩放因子，默认为 IMAGE_FACTOR
    :param min_pixels: 图像最小像素数，默认为 MIN_PIXELS
    :param max_pixels: 图像最大像素数，默认为 MAX_PIXELS
    :return: 缩放后的高度和宽度的元组
    """
    # 检查图像的宽高比是否超过最大允许比例
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"绝对宽高比必须小于 {MAX_RATIO}，当前为 {max(height, width) / min(height, width)}"
        )
    # 对高度进行四舍五入，使其为 factor 的倍数，并确保不小于 factor
    h_bar = max(factor, round_by_factor(height, factor))
    # 对宽度进行四舍五入，使其为 factor 的倍数，并确保不小于 factor
    w_bar = max(factor, round_by_factor(width, factor))
    # 如果缩放后的像素数超过最大像素数
    if h_bar * w_bar > max_pixels:
        # 计算缩放比例
        beta = math.sqrt((height * width) / max_pixels)
        # 对高度进行向下取整，使其为 factor 的倍数
        h_bar = floor_by_factor(height / beta, factor)
        # 对宽度进行向下取整，使其为 factor 的倍数
        w_bar = floor_by_factor(width / beta, factor)
    # 如果缩放后的像素数小于最小像素数
    elif h_bar * w_bar < min_pixels:
        # 计算缩放比例
        beta = math.sqrt(min_pixels / (height * width))
        # 对高度进行向上取整，使其为 factor 的倍数
        h_bar = ceil_by_factor(height * beta, factor)
        # 对宽度进行向上取整，使其为 factor 的倍数
        w_bar = ceil_by_factor(width * beta, factor)
    
    # 再次检查宽高比，确保不超过最大允许比例
    if h_bar / w_bar > MAX_RATIO:
        h_bar = w_bar * MAX_RATIO    
    elif w_bar / h_bar > MAX_RATIO:
        w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    """
    从字典中获取图像并进行缩放处理。
    :param ele: 包含图像信息的字典，可能包含 'image'（PIL 图像对象）、'image_url'（图像 URL）、'resized_height' 和 'resized_width' 等键
    :param size_factor: 缩放因子，默认为 IMAGE_FACTOR
    :return: 处理后的 PIL 图像对象
    """
    # 尝试从字典中获取图像数据
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    # 如果图像是 PIL 图像对象
    if isinstance(image, Image.Image):
        image_obj = image
    # 如果图像是 HTTP 或 HTTPS URL
    elif image.startswith("http://") or image.startswith("https://"):
        # 发送请求并打开图像
        image_obj = Image.open(requests.get(image, stream=True).raw)
    # 如果图像是本地文件 URL
    elif image.startswith("file://"):
        # 打开本地文件
        image_obj = Image.open(image[7:])
    # 如果图像是 Base64 编码的字符串
    elif image.startswith("data:image"):
        # 提取 Base64 数据
        data = image.split(";", 1)[1]
        if data.startswith("base64,"):
            # 解码 Base64 数据
            data = base64.b64decode(data[7:])
            # 打开图像
            image_obj = Image.open(BytesIO(data))
    else:
        # 尝试打开本地文件
        image_obj = Image.open(image)
    # 如果未成功获取图像对象，抛出错误
    if image_obj is None:
        raise ValueError(f"无法识别的图像输入，支持本地路径、HTTP URL、Base64 和 PIL.Image，当前输入为 {image}")
    # 将图像转换为 RGB 模式
    image = image_obj.convert("RGB")
    ## 进行图像缩放
    if "resized_height" in ele and "resized_width" in ele:
        # 如果字典中提供了缩放后的高度和宽度，使用这些值进行缩放
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        # 获取图像的原始宽度和高度
        width, height = image.size
        # 获取最小像素数，默认为 MIN_PIXELS
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        # 获取最大像素数，默认为 MAX_PIXELS
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        # 调用 smart_resize 函数进行缩放
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    # 对图像进行缩放
    image = image.resize((resized_width, resized_height))

    return image


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image]:
    """
    从给定的字典中提取视频信息，对视频进行处理并返回处理后的视频数据。
    处理过程包括视频帧的选择、尺寸调整等。
    参数:
    ele (dict): 包含视频信息的字典，可能包含 'video'（视频文件路径）、
                'video_start'（视频开始时间）、'video_end'（视频结束时间）等键。
    image_factor (int, 可选): 图像缩放因子，默认为 IMAGE_FACTOR。

    返回:
    torch.Tensor | list[Image.Image]: 如果输入是视频文件路径，返回处理后的视频张量；
                                      如果输入是视频帧列表，返回处理后的图像列表。
    """
    if isinstance(ele["video"], str):
        # TODO: support http url
        # 若 'video' 键的值是字符串，意味着它可能是视频文件路径
        video = ele["video"]
        # 若路径以 'file://' 开头，去除该前缀
        if video.startswith("file://"):
            video = video[7:]

        # 利用 torchvision 的 io.read_video 函数读取视频
        # start_pts 和 end_pts 分别表示视频的起始和结束时间点，单位为秒
        # output_format="TCHW" 表示输出的视频张量维度为 (帧数, 通道数, 高度, 宽度)
        video, audio, info = io.read_video(
            video,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )

        # 确保输入中不同时包含 'fps' 和 'nframes' 键
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            # 若存在 'nframes' 键，将其值按 FRAME_FACTOR 取整
            nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
        else:
            # 若不存在 'nframes' 键，根据帧率和其他参数计算帧数
            fps = ele.get("fps", FPS)
            # 最小帧数，按 FRAME_FACTOR 向上取整
            min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
            # 最大帧数，按 FRAME_FACTOR 向下取整
            max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, video.size(0))), FRAME_FACTOR)
            # 计算帧数
            nframes = video.size(0) / info["video_fps"] * fps
            # 确保帧数在最小和最大帧数之间
            nframes = min(max(nframes, min_frames), max_frames)
            # 按 FRAME_FACTOR 取整
            nframes = round_by_factor(nframes, FRAME_FACTOR)
        # 确保帧数在有效范围内
        if not (FRAME_FACTOR <= nframes and nframes <= video.size(0)):
            raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {video.size(0)}], but got {nframes}.")

        # 生成用于选择视频帧的索引
        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        # 获取视频的高度和宽度
        height, width = video.shape[2:]
        # 根据索引选择视频帧
        video = video[idx]

        # 获取最小像素数，默认为 VIDEO_MIN_PIXELS
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        # 获取总像素数，默认为 VIDEO_TOTAL_PIXELS
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        # 计算最大像素数
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        # 若存在 'max_pixels' 键，使用该值
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            # 若存在 'resized_height' 和 'resized_width' 键，使用这些值进行尺寸调整
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            # 若不存在，根据原始高度和宽度进行尺寸调整
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        # 对视频进行尺寸调整
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video
    else:
        # 若 'video' 键的值不是字符串，假设它是视频帧列表
        assert isinstance(ele["video"], (list, tuple))
        # 复制字典并去除 'type' 和 'video' 键
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        # 对视频帧列表中的每一帧进行处理
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        # 计算帧数，按 FRAME_FACTOR 向上取整
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        # 若帧数不足，用最后一帧填充
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    """
    从对话列表中提取视觉信息（图像或视频信息）。

    参数:
    conversations (list[dict] | list[list[dict]]): 对话列表，可能是单层或嵌套列表。

    返回:
    list[dict]: 包含视觉信息的字典列表。
    """
    vision_infos = []
    # 若 conversations 是单层列表，将其转换为嵌套列表
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    # 遍历每个对话
    for conversation in conversations:
        # 遍历对话中的每条消息
        for message in conversation:
            if isinstance(message["content"], list):
                # 若消息内容是列表，遍历列表中的每个元素
                for ele in message["content"]:
                    # 若元素包含图像或视频相关信息，将其添加到视觉信息列表中
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    """
    处理对话中的视觉信息，提取图像和视频输入。

    参数:
    conversations (list[dict] | list[list[dict]]): 对话列表，可能是单层或嵌套列表。

    返回:
    tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
        包含图像输入列表和视频输入列表的元组，若没有相应输入则为 None。
    """
    # 提取视觉信息
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    # 遍历视觉信息
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            # 若包含图像信息，使用 fetch_image 函数处理并添加到图像输入列表中
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            # 若包含视频信息，使用 fetch_video 函数处理并添加到视频输入列表中
            video_inputs.append(fetch_video(vision_info))
        else:
            # 若不包含图像或视频信息，抛出错误
            raise ValueError("image, image_url or video should in content.")
    # 若图像输入列表为空，将其设为 None
    if len(image_inputs) == 0:
        image_inputs = None
    # 若视频输入列表为空，将其设为 None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs



def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor:
    """
    从给定的字典中提取视频信息，对视频进行处理并返回处理后的视频张量。
    处理过程包括视频帧的选择、尺寸调整等。
    参数:
    ele (dict): 包含视频信息的字典，可能包含 'video'（视频文件路径或帧列表）、
                'video_start'（视频开始时间）、'video_end'（视频结束时间）等键。
    image_factor (int, 可选): 图像缩放因子，默认为 IMAGE_FACTOR。
    返回:
    torch.Tensor: 处理后的视频张量，形状为 [帧数, 通道数, 高度, 宽度]
    """
    if isinstance(ele["video"], str):
        # 处理视频文件路径
        video = ele["video"]
        if video.startswith("file://"):
            video = video[7:]
        video, audio, info = io.read_video(
            video,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        # 帧数计算
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
        else:
            fps = ele.get("fps", FPS)
            min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
            max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, video.size(0))), FRAME_FACTOR)
            nframes = video.size(0) / info["video_fps"] * fps
            nframes = min(max(nframes, min_frames), max_frames)
            nframes = round_by_factor(nframes, FRAME_FACTOR)

        if not (FRAME_FACTOR <= nframes and nframes <= video.size(0)):
            raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {video.size(0)}], but got {nframes}.")

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        height, width = video.shape[2:]
        video = video[idx] # 选择指定的帧数

        # 视频尺寸调整
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        
        return video

    else:
        # 处理图像帧列表
        assert isinstance(ele["video"], (list, tuple)), "Input must be a list/tuple of frames or a video path"
        # 复制字典并去除不相关键
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        
        # 处理每一帧图像
        images = [
            fetch_image({"image": frame, **process_info}, size_factor=image_factor)
            for frame in ele["video"]
        ]
        
        # 计算目标帧数
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        
        # 填充帧列表以达到目标长度
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        
        # 将PIL.Image列表转换为张量 [帧数, 通道数, 高度, 宽度]
        video_tensor = torch.stack([transforms.ToTensor()(img) for img in images])
        
        # 调整尺寸
        _, _, height, width = video_tensor.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            
        video_tensor = transforms.functional.resize(
            video_tensor,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video_tensor


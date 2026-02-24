from itertools import repeat
from torch.jit import isinstance
import logging
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments
import torch
from src.model_utils import LLAVA_NEXT, QWEN2_VL, PHI3V, process_vlm_inputs_fns

logger = logging.getLogger(__name__)

# 定义常量
PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)  # Phi模型图像token的最大ID
LLAVA_IMAGE_TOKEN_ID = 32000  # LLaVA模型图像token的ID

def process_vlm_inputs(model_inputs: dict, processor, backbone_name, max_length=None):
    """
    处理视觉语言模型的输入数据, 支持多种不同的VLM架构
    
    参数:
        model_inputs: 包含文本和图像的字典 {'text': [...], 'image': [...]}
        processor: 用于处理输入数据的处理器
        backbone_name: 模型架构名称 (LLAVA_NEXT, QWEN2_VL, PHI3V)
        max_length: 最大序列长度
        
    返回:
        包含处理后的输入数据的字典
    """
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False  # 标记批次中是否存在图像
    
    # 1. 遍历每个文本-图像对并进行处理(因为处理器不支持批量处理)
    for text, image in zip(texts, images):
        if image is None:
            # 处理没有图像的情况
            if backbone_name == LLAVA_NEXT:
                inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
            elif backbone_name == QWEN2_VL:
                inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            elif backbone_name == PHI3V:
                inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            
            # 处理输入ID
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # 如果是空字符串，只包含BOS token
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            # 处理有图像的情况
            image_exists = True
            if backbone_name == LLAVA_NEXT:
                inputs = processor(images=image, text=text, return_tensors="np", max_length=max_length, truncation=True)
            elif backbone_name == QWEN2_VL:
                inputs = processor(images=[image], text=[text], return_tensors="np", max_length=max_length, truncation=True)
            elif backbone_name == PHI3V:
                inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. 对输入进行填充
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    
    # 3. 处理混合批次(同一批次中有些样本有图像，有些没有)
    if image_exists:
        if backbone_name == LLAVA_NEXT:
            # 基于第一个有效数据点创建虚拟图像输入
            pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
            image_size_for_padding = torch.from_numpy(list(v for v in image_sizes if v is not None)[0])
            
            # 创建完整的张量
            pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            image_sizes = [torch.from_numpy(v) if v is not None else image_size_for_padding for v in image_sizes]
            image_sizes = torch.cat(image_sizes, dim=0)
            
        if backbone_name == QWEN2_VL:
            pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
            pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            
            if image_grid_thw:
                image_grid_thw_for_padding = torch.from_numpy(list(v for v in image_grid_thw if v is not None)[0])
                image_grid_thw = [torch.from_numpy(v) if v is not None else image_grid_thw_for_padding for v in image_grid_thw]
                image_grid_thw = torch.cat(image_grid_thw, dim=0)
                inputs['image_grid_thw'] = image_grid_thw
        
        # 将处理后的图像数据添加到输入中
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        # 如果没有图像，创建虚拟图像数据
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs



def split_dense_inputs(model_input: dict, chunk_size: int):
    """
    分割密集输入数据
    
    参数:
        model_input: 输入数据字典
        chunk_size: 每个块的大小
        
    返回:
        分割后的输入数据列表
    """
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]

def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    """
    分割并处理VLM输入数据
    
    参数:
        model_input: 输入数据字典
        chunk_size: 每个块的大小
        
    返回:
        分割后的输入数据列表
    """
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = []
    for k in keys:
        if isinstance(arg_val[k], torch.Tensor):
            chunked_tensor = arg_val[k].split(chunk_size, dim=0)
        else:
            chunked_tensor = [arg_val[k][i: i + chunk_size] for i in list(range(0, len(arg_val[k]), chunk_size))]
        chunked_tensors.append(chunked_tensor)
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
    chunked_inputs = [{arg_key: c} for c in chunked_arg_val]

    return chunked_inputs

def split_vlm_inputs(model_input: dict, chunk_size: int):
    """
    分割VLM输入数据，考虑图像token的位置
    
    参数:
        model_input: 输入数据字典
        chunk_size: 每个块的大小
        
    返回:
        分割后的输入数据列表
    """
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]
    keys = list(arg_val.keys())

    # 直接分割input_ids和attention_mask
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in ["input_ids", "attention_mask"]]

    # 根据图像token的位置分割pixel_values和image_sizes
    input_ids = arg_val["input_ids"]
    positions = torch.nonzero((input_ids < 0) & (input_ids > -PHI_IMAGE_TOKEN_MAX_INPUT_ID), as_tuple=True)
    row_contain_image = torch.unique(positions[0])  # 标记哪些行包含图像
    
    # 计算每个块中的图像数量
    num_chunks = len(chunked_tensors[0])
    chunk_image_count = []
    for chunk_idx in range(num_chunks):
        chunk_image_count.append(torch.sum(
            (row_contain_image >= chunk_idx * chunk_size) & (row_contain_image < (chunk_idx + 1) * chunk_size)).item())
    
    # 处理图像数据
    if "pixel_values" in keys:
        pixel_values = arg_val["pixel_values"]
        image_sizes = arg_val["image_sizes"]
        chunked_tensors.append(torch.split(pixel_values, chunk_image_count))
        chunked_tensors.append(torch.split(image_sizes, chunk_image_count))

    # 组装分割后的数据
    chunked_arg_val = []
    for kk, tt in zip(repeat(keys), zip(*chunked_tensors)):
        if "pixel_values" in keys and tt[2].numel() == 0:  # 这个块不包含图像
            chunked_arg_val.append(dict(zip(kk[:2], tt[:2])))
        else:
            chunked_arg_val.append(dict(zip(kk, tt)))

    return [{arg_key: c} for c in chunked_arg_val]

def get_dense_rep(x):
    """
    获取密集表示(qry_reps或tgt_reps)
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    else:
        return x["qry_reps"]
    

@dataclass
class TrainTextImageDataCollator:
    """
    训练数据的收集器，处理查询文本/图像和正样本文本/图像
    
    参数:
        data_args: 数据参数
        model_args: 模型参数
        processor: 数据处理器
    """
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        处理一批训练样本
        
        参数:
            examples: 包含查询文本、查询图像、正样本文本、正样本图像的列表
            
        返回:
            处理后的查询输入和正样本输入
        """
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        neg_inputs = self._get_batch_inputs(examples, "neg_text", "neg_image")
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_keyname, image_keyname):
        """
        获取批处理输入
        
        参数:
            examples: 样本列表
            text_keyname: 文本键名
            image_keyname: 图像键名
            
        返回:
            包含文本和图像的字典
        """
        texts, images = [], []
        for example in examples:
            # 处理无效数据示例(使用虚拟输入)
            if example is None or not example:
                text, image = '  ', None
            text, image = example[text_keyname], example[image_keyname]
            if type(text) == list:
                if len(text) == 0 or len(image) == 0:
                    text, image = '  ', None
                else:
                    text, image = text[0], image[0]
            texts.append(text)
            images.append(image)
        inputs = {'text': texts, 'image': images}
        return inputs
    


@dataclass
class EvalCollator:
    """
    评估数据的收集器
    
    参数:
        data_args: 数据参数
        model_args: 模型参数
        processor: 数据处理器
    """
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        处理一批评估样本
        
        参数:
            examples: 包含文本和图像的列表
            
        返回:
            处理后的输入数据
        """
        examples = {'text': [e[0] for e in examples], 'image': [e[1] for e in examples]}
        inputs = process_vlm_inputs_fns[self.model_args.model_backbone](
            examples,
            processor=self.processor,
            max_length=self.data_args.max_len
        )
        return inputs
    


@dataclass
class OpenCLIPCollator:
    """
    OpenCLIP模型的数据收集器
    
    参数:
        data_args: 数据参数
        vis_processors: 视觉处理器
        txt_processors: 文本处理器
    """
    data_args: DataArguments
    vis_processors: AutoProcessor
    txt_processors: AutoTokenizer

    def __call__(self, examples):
        """
        处理一批OpenCLIP样本
        
        参数:
            examples: 包含文本和图像的列表
            
        返回:
            处理后的输入数据
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        """
        获取批处理输入
        
        参数:
            examples: 样本列表
            
        返回:
            包含处理后的输入数据的字典
        """
        input_ids, pixel_values, attention_mask = [], [], []
        image_exist, text_exist = False, False
        for example in examples:
            text, image = example
            if image is not None:
                if image.mode == 'L':  # 如果是灰度图像，转换为RGB
                    image = image.convert('RGB')
                image_inputs = self.vis_processors(image).unsqueeze(0)
                image_exist = True
                pixel_values.append(image_inputs)
            if text:
                text_exist = True
            text_inputs = self.txt_processors(text)
            input_ids.append(text_inputs)
            
        # 处理文本输入
        if text_exist:
            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = input_ids.ne(0)
            
        # 处理图像输入
        if image_exist:
            pixel_values = torch.cat(pixel_values, dim=0)
            
        # 验证文本和图像数量是否匹配
        if text_exist and image_exist:
            assert input_ids.size()[0] == pixel_values.size()[0]
            
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }

        return inputs
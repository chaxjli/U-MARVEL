from typing import Dict, Sequence
import torch

from . import register_collator
from .base import BaseDataCollator

# 从当前包的 qwen2_vision_process 模块导入 process_vision_info 函数
from .qwen2_vision_process import process_vision_info

# 使用 register_collator 装饰器将该类注册为 "qwen2-vl-7b" 类型的数据收集器
@register_collator("qwen2-vl-7b")
class Qwen2VL7BDataCollator(BaseDataCollator):
    
    # 定义一个属性 PAD_TOKEN_ID，用于获取分词器的填充标记的 ID
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    # 定义 __call__ 方法，使得该类的实例可以像函数一样被调用
    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        # 获取 messages 列表中第一个元素的键的数量，用于判断是否有硬负样本
        category_size = len(messages[0])
        
        # 如果键的数量为 3，则认为存在硬负样本
        if category_size == 3:
            has_hard_negative = True
        else:
            has_hard_negative = False

        # 初始化一个空列表，用于存储重新组织后的消息
        new_messages = []
        
        # 遍历每个类别
        for category in range(category_size):
            # 遍历 messages 列表中的每个元素
            for item in messages:
                # 将每个元素中对应类别的消息添加到 new_messages 列表中
                new_messages.append(item[category])

        # 对 new_messages 列表中的每个消息应用聊天模板，不进行分词，不添加生成提示
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in new_messages
        ]
        # 调用 process_vision_info 函数处理视觉信息，返回图像输入和视频输入
        image_inputs, video_inputs = process_vision_info(new_messages)
        # 使用处理器对文本、图像和视频进行处理，进行填充操作，并返回 PyTorch 张量
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 从处理后的输入中获取输入 ID
        input_ids = inputs['input_ids']
        
        # 复制输入 ID 作为标签
        labels = input_ids.clone()

        # 将标签中填充标记的 ID 替换为忽略标记的 ID
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        # 检查处理后的输入中是否包含注意力掩码
        if 'attention_mask' in inputs:
            # 如果包含，则获取注意力掩码
            attention_mask = inputs['attention_mask']
        else:
            # 如果不包含，则将注意力掩码设置为 None
            attention_mask = None
        
        # 检查处理后的输入中是否包含像素值
        if 'pixel_values' in inputs:
            # 如果包含，则获取像素值
            pixel_values = inputs['pixel_values']
        else:
            # 如果不包含，则将像素值设置为 None
            pixel_values = None
        
        # 检查处理后的输入中是否包含图像网格信息
        if 'image_grid_thw' in inputs:
            # 如果包含，则获取图像网格信息
            image_grid_thw = inputs['image_grid_thw']
        else:
            # 如果不包含，则将图像网格信息设置为 None
            image_grid_thw = None

        # 返回一个字典，包含输入 ID、注意力掩码、像素值、图像网格信息、标签和是否有硬负样本的标志
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative
        )
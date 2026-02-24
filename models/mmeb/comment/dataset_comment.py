from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os

from torch.jit import isinstance  # 注意：此处可能应为 torch.isinstance

from src.model_utils import PHI3V, vlm_image_tokens  # 模型相关常量与工具
from src.utils import print_master, print_rank  # 分布式打印工具

# from datasets import Dataset  # @ruimeng, still buggy（可能存在问题的导入）
from torch.utils.data import Dataset


def process_image(image, resolution, max_dim=1344):
    """
    处理图像分辨率，支持三种预设分辨率或自适应调整
    - high: 1344x1344
    - mid: 672x672
    - low: 128x128
    - 其他: 保持原始比例，最长边不超过max_dim
    """
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image
# torchrun --nproc_per_node=2 --master_port=22447 --max_restarts=0 train.py \
#  --model_name microsoft/Phi-3.5-vision-instruct --bf16 --pooling last \
#  --model_backbone phi3_v \
#  --dataset_name TIGER-Lab/MMEB-train \
#  --subset_name ImageNet_1K N24News HatefulMemes InfographicsVQA ChartQA Visual7W VisDial CIRR NIGHTS WebQA MSCOCO \
#  --num_sample_per_subset 50000 \
#  --image_dir MMEB-train \
#  --max_len 256 --num_crops 4 --output_dir $OUTPUT_DIR --logging_steps 1 \
#  --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 2000 \
#  --warmup_steps 200 --save_steps 1000 --normalize True \
#  --temperature 0.02 --per_device_train_batch_size 8 \
#  --grad_cache True --gc_q_chunk_size 2 --gc_p_chunk_size 2 
class TrainTextImageDataset(Dataset):
    """
    训练阶段的图像-文本对数据集，支持正负样本三元组
    """
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        train_data = []
        print_rank(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        
        # 加载多个子集数据
        # # 数据集分割名称，支持原始分割或多样化指令分割
        # split_name: default='original' 否则指定:  'diverse_instruction'"
        for subset in data_args.subset_name:
            subset_data = load_dataset(self.data_args.dataset_name, subset, split=data_args.split_name)
            train_data.append(subset_data[0])  # 注意：此处可能应为 append(subset_data)
        self.train_data = concatenate_datasets(train_data)  # 合并所有子集

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, img_path):
        """加载并预处理图像，支持不同模型的分辨率要求"""
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def __getitem__(self, data_idx) -> Tuple[str, List[str]]:
        """
        获取索引对应的数据样本，包含：
        - 查询文本与图像
        - 正例文本与图像
        - 负例文本与图像（如果存在）
        """
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )
        
        # 处理负样本（可选）
        if 'neg_text' in self.train_data.column_names:
            neg_texts, neg_image_paths = self.train_data[data_idx]["neg_text"], self.train_data[data_idx]["neg_image_path"]
        else:
            # 若没有负样本，创建空列表
            neg_texts, neg_image_paths = [''] * len(data_idx), [] * len(data_idx)
        
        # 确保数据为列表格式（支持单样本或批量）
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            neg_texts = [neg_texts]
            neg_image_paths = [neg_image_paths]
        
        # 处理每个样本对
        _qry_texts, _qry_images, _pos_texts, _pos_images, _neg_texts, _neg_images = [], [], [], [], [], []
        backbone = self.model_args.model_backbone
        for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path \
            in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths, neg_texts, neg_image_paths):
            
            # 根据模型类型替换图像特殊令牌（如<|image_1|>）
            if backbone != PHI3V:
                qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                neg_text = neg_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) if neg_text else None
            
            # 加载图像
            qry_image = self._get_image(qry_image_path)
            pos_image = self._get_image(pos_image_path)
            neg_image = self._get_image(neg_image_path) if neg_image_path else None
            
            # 过滤无效样本
            if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                print("empty inputs")
                continue
                
            _qry_texts.append(qry_text)
            _qry_images.append(qry_image)
            _pos_texts.append(pos_text)
            _pos_images.append(pos_image)
            _neg_texts.append(neg_text)
            _neg_images.append(neg_image)

        return {"query_text": _qry_texts, "query_image": _qry_images,
                "pos_text": _pos_texts, "pos_image": _pos_images,
                "neg_text": _neg_texts, "neg_image": _neg_images}
    
# python eval.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path outputs/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name N24News CIFAR-100 HatefulMemes VOC2007 SUN397 ImageNet-A ImageNet-R ObjectNet Country211 \
#   --dataset_split test --per_device_eval_batch_size 16 \
#   --image_dir eval_images/

class EvalDataset(Dataset):
    """
    评估阶段的数据集，支持文本-图像检索任务
    功能：加载评估数据、去重文本-图像对、预处理图像和文本
    """
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        """
        参数:
            data_args (DataArguments): 数据配置参数（含dataset_name、image_dir等）
            model_args (ModelArguments): 模型配置参数（含model_backbone）
            subset (str): 数据集子集名称（如'diverse_instruction')
            text_field (str): 数据中表示文本的字段名（如'caption'）
            img_path_field (str): 数据中表示图像路径的字段名（如'image_path'）
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone  # 获取模型骨干类型（如LLAVA_NEXT、Phi3V）

        # 加载评估数据集（从Hugging Face或本地）
        self.eval_data = load_dataset(
            self.data_args.dataset_name,  # 数据集名称（如'flickr30k'）
            subset,  # 子集名称（可选）
            split=self.data_args.dataset_split,  # 数据集分割（如'test'）
        )
        
        # 处理原始数据，生成唯一的文本-图像对（去重）
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        
        # 将去重后的对转换为Hugging Face Dataset格式
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],  # 所有文本
            "img_path": [pair["img_path"] for pair in self.paired_data]  # 所有图像路径
        })
    def __len__(self):
        return len(self.paired_dataset)  # 返回去重后的样本总数
    def __getitem__(self, item):
        """
        获取单个评估样本，包含文本和预处理后的图像
        
        参数:
            item (int): 样本索引
        
        返回:
            tuple: (文本, 图像Tensor)
        """
        # 从paired_dataset中获取文本和图像路径
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        
        # 适配不同模型的图像标记（如将Phi3V的图像标记替换为当前模型的标记）
        if self.backbone != PHI3V:
            # vlm_image_tokens是模型对应的图像标记（如<image>）
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.backbone])
        
        # 加载并预处理图像
        return text, self._get_image(img_path),  # 注意逗号：返回元组（支持PyTorch DataLoader）
    def _process_image(self, image, resolution):
        """
        简化的图像预处理函数，调整图像分辨率
        
        参数:
            image (PIL.Image): 原始图像对象
            resolution (str): 分辨率模式（'high'或默认）
        
        返回:
            PIL.Image: 调整大小后的图像
        """
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))  # 高分辨率（如LLaVA-Next使用）
        else:
            image = image.resize((336, 336))  # 低分辨率（默认）
        return image
    def _get_image(self, img_path):
        """
        加载图像文件并进行预处理
        
        参数:
            img_path (str): 图像文件相对路径
        
        返回:
            PIL.Image: 预处理后的图像对象
        """
        if img_path == "":  # 空路径处理
            return None
        # 拼接完整路径（基于data_args.image_dir）
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path).convert("RGB")  # 打开图像并转为RGB格式
        
        # 根据模型骨干和配置进行预处理
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return self._process_image(image, self.data_args.image_resolution)
        else:
            return image  # Phi3V可能使用默认预处理或其他方式
        
        # 注意：最后一行return image可能冗余，前两个分支已覆盖所有情况
    def get_paired_data(self, text_field, img_path_field):
        """
        处理原始数据，生成唯一的文本-图像对，支持多种数据格式
        
        参数:
            text_field (str): 文本字段名
            img_path_field (str): 图像路径字段名
        
        返回:
            list: 去重后的文本-图像对列表（字典形式）
        """
        unique_pair = set()  # 使用集合去重（元素为元组(text, img_path)）
        
        for row in self.eval_data:  # 遍历原始数据的每一行
            # 处理文本字段为字符串的情况（单文本对应单图像或多图像）
            if isinstance(row[text_field], str):
                if row[text_field]:  # 非空文本
                    # 添加文本-图像对（图像路径可能是单个字符串或列表）
                    if isinstance(row[img_path_field], List):
                        # 文本对应多个图像路径（如同一文本描述多张图）
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        # 文本对应单个图像路径
                        unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    # 空文本处理（仅添加图像路径，可能用于图像检索文本的场景）
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add(("", img_path))
                    else:
                        unique_pair.add(("", row[img_path_field]))
            # 处理文本字段为列表的情况（多文本对应多图像，需一一对应）
            elif type(row[text_field]) == list:
                assert type(row[img_path_field]) == list and len(row[img_path_field]) == len(row[text_field]), \
                    "文本和图像列表长度必须一致"
                # 逐个配对文本和图像路径
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))
        
        # 将集合转换为列表，每个元素为字典（便于后续处理）
        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data

    




class FlickrDataset(Dataset):
    """
    Flickr 1K 测试数据集，专门用于图像-文本检索评估
    支持两种模态：
    - image: 图像到文本检索（给定图像，检索描述文本）
    - text: 文本到图像检索（给定文本，检索匹配图像）
    """
    def __init__(self, modality, model_backbone):
        self.model_backbone = model_backbone
        self.modality = modality
        self.raw_data = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split="test")
        
        # 根据模态准备数据
        if modality == "image":
            self.eval_data, self.image_names = self.get_image_data()  # 图像到文本
        else:
            self.eval_data, self.image_names = self.get_text_data()  # 文本到图像

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        """获取单个评估样本"""
        text, image = self.eval_data[idx]
        if self.model_backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_backbone])
            if self.data_args.image_resolution:  # 注意：此处可能存在未定义的 data_args
                image = process_image(image, self.data_args.image_resolution)
        return text, image

    def get_image_data(self):
        """准备图像到文本检索的数据（图像作为查询，文本作为候选）"""
        eval_data, image_names = [], []
        # 构建查询指令（提示模型生成图像描述）
        inst = "<|image_1|> Find an image caption describing the given image."
        
        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
            image_names.append(row["filename"])
        return eval_data, image_names

    def get_text_data(self):
        """准备文本到图像检索的数据（文本作为查询，图像作为候选）"""
        eval_data, image_names = [], []
        inst = ""  # 文本查询的前缀（可根据任务调整）
        
        for row in self.raw_data:
            for caption in row["caption"]:
                eval_data.append((inst + caption, None))  # 文本作为查询，无图像
                image_names.append(row["filename"])
        return eval_data, image_names
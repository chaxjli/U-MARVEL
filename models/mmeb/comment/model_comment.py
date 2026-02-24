from typing import Dict, Optional
import torch
import torch.distributed as dist  # 分布式训练工具
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig  # Hugging Face模型相关
from peft import LoraConfig, get_peft_model, PeftModel  # 参数高效微调（LoRA）
from src.arguments import ModelArguments, TrainingArguments  # 自定义参数类
from src.model_utils import (
    LLAVA_NEXT, QWEN2_VL, PHI3V,  # 模型类型常量
    get_backbone_name, print_master,  # 工具函数
    QWEN2_5_VL, backbone2model  # 模型映射关系
)
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM  # 自定义Phi3V模型
from src.vlm_backbone.llava_next import LlavaNextForConditionalGeneration  # 自定义LLaVA-Next模型
class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM  # 默认因果语言模型类
    def __init__(self,
             encoder: PreTrainedModel,  # 基础编码器模型
             pooling: str = 'cls',  # 池化方法（'cls'或自定义）
             normalize: bool = False,  # 是否归一化特征向量
             temperature: float = 1.0,  # 相似性计算温度参数
                 ):
        super().__init__()
        self.config = encoder.config  # 保存模型配置
        self.encoder = encoder  # 基础模型（如LLaVA-Next、Phi3V）
        self.pooling = pooling  # 池化策略
        self.normalize = normalize  # 归一化标志
        self.temperature = temperature  # 温度参数，控制softmax平滑度
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')  # 交叉熵损失函数
        self.is_ddp = dist.is_initialized()  # 是否处于分布式训练环境
        if self.is_ddp:
            self.process_rank = dist.get_rank()  # 当前进程ID
            self.world_size = dist.get_world_size()  # 总进程数（GPU数）
    def encode_input(self, input: Dict[str, Tensor]):
        """
        对输入数据（文本/图像）进行编码，返回池化后的特征向量。
        
        参数:
            input (Dict[str, Tensor]): 输入数据（如token_ids、attention_mask、images等）
        
        返回:
            Tensor: 池化后的特征向量（形状为 [batch_size, hidden_dim]）
        """
        # 使用基础模型编码输入，返回隐藏状态和注意力掩码
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        last_hidden_state = hidden_states.hidden_states[-1]  # 取最后一层隐藏状态
        pooled_output = self._pooling(last_hidden_state, input['attention_mask'])  # 应用池化
        return pooled_output
    def _pooling(self, last_hidden_state: Tensor, attention_mask: Tensor):
        """
        根据池化策略提取特征向量。
        
        参数:
            last_hidden_state (Tensor): 最后一层隐藏状态（形状 [batch, seq_len, hidden_dim]）
            attention_mask (Tensor): 注意力掩码（形状 [batch, seq_len]）
        
        返回:
            Tensor: 池化后的特征向量（形状 [batch, hidden_dim]）
        """
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])  # 判断是否为左填充
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # 左填充时，取每个样本的最后一个token的隐藏状态
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # 非左填充时，根据注意力掩码计算真实末尾位置（EOS）
                eos_indices = attention_mask.sum(dim=1) - 1  # 每个样本的最后一个有效token索引
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices
                ]
        else:
            raise NotImplementedError(f"Pooling method {self.pooling} not supported")
        
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)  # L2归一化特征向量
        return reps
    @classmethod
    def build(cls, model_args: ModelArguments, training_args: TrainingArguments=None, **kwargs):
        """
        构建MMEBModel实例，支持不同模型架构（LLaVA-Next、Phi3V、Qwen-VL等）和LoRA微调。
        
        参数:
            model_args (ModelArguments): 模型配置参数
            training_args (TrainingArguments): 训练配置参数（可选）
            **kwargs: 其他传递给模型加载的参数
        
        返回:
            MMEBModel: 初始化后的模型实例
        """
        # 从预训练路径加载模型配置
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)  # 获取模型骨干类型（如LLAVA_NEXT）
        setattr(model_args, 'model_backbone', model_backbone)  # 将骨干类型保存到配置中
        print_master(f'Loading backbone [{model_backbone}]')  # 主进程日志：打印骨干类型

        # 根据骨干类型加载不同的基础模型
        if model_backbone == PHI3V:
            # 配置Phi3V模型参数
            config._attn_implementation = "eager"  # 注意力实现方式
            config.padding_side = "right"  # 右填充
            config.use_cache = False  # 禁用推理缓存（训练时常用）
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,  # BF16精度
                low_cpu_mem_usage=True,  # 低内存占用模式
            )
        elif model_backbone == LLAVA_NEXT:
            # 配置LLaVA-Next模型参数
            config.use_cache = False
            config.padding_side = "left"  # 左填充
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            # 配置Qwen-VL模型参数
            config._attn_implementation = "flash_attention_2"  # 使用FlashAttention 2加速
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(  # 通过映射获取模型类
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            # 通用Transformer模型（如BLOOM、GPT系列）
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name,
                **kwargs,
                config=config,
                attn_implementation="flash_attention_2",  # 使用FlashAttention加速
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,  # 信任远程代码（用于自定义模型）
            )

        # 应用LoRA微调（如果启用）
        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,  # LoRA秩参数
                lora_alpha=model_args.lora_alpha,  # 缩放参数
                target_modules=model_args.lora_target_modules.split(','),  # 目标模块列表（逗号分隔）
                lora_dropout=model_args.lora_dropout,  # Dropout率
                init_lora_weights="gaussian",  # 初始化方式
                use_dora=True,  # 启用DORA优化（可选）
                inference_mode=False,  # 训练模式
            )
            lora_model = get_peft_model(base_model, lora_config)  # 将LoRA应用到基础模型
            model = cls(
                encoder=lora_model,  # 使用LoRA模型作为编码器
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        else:
            # 不使用LoRA，直接使用基础模型
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        return model
    @classmethod
    def load(cls, model_args: ModelArguments, **kwargs):
        """
        从检查点加载预训练模型（支持LoRA合并）。
        
        参数:
            model_args (ModelArguments): 模型配置参数（需包含checkpoint_path）
            **kwargs: 其他传递给模型加载的参数
        
        返回:
            MMEBModel: 加载后的模型实例
        """
        checkpoint_path = model_args.checkpoint_path or model_args.model_name  # 检查点路径
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')

        # 根据骨干类型加载基础模型（与build方法类似）
        if model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL}:
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"  # 视觉模块注意力配置
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                config=config,
            )
        elif model_backbone == PHI3V:
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                **kwargs,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            base_model.padding_side = "right"
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                checkpoint_path,
                **kwargs,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        # 处理LoRA模型加载（从检查点合并权重）
        if model_args.lora:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)  # 从检查点加载LoRA配置
            lora_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=lora_config)
            merged_model = lora_model.merge_and_unload()  # 合并LoRA权重并卸载适配器
            model = cls(
                encoder=merged_model,  # 使用合并后的模型作为编码器
                pooling=model_args.pooling,
                normalize=model_args.normalize,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
            )
        return model
    def save(self, output_dir: str):
        """保存模型编码器到指定目录。"""
        self.encoder.save_pretrained(output_dir)  # 调用Hugging Face模型保存接口
    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        """
        计算查询和目标的相似性损失（适用于对比学习任务）。
        
        参数:
            qry (Dict[str, Tensor]): 查询输入（如文本/图像特征）
            tgt (Dict[str, Tensor]): 目标输入（如文本/图像特征）
        
        返回:
            Dict: 包含损失（loss）、查询特征（qry_reps）、目标特征（tgt_reps）
        """
        # 编码查询和目标输入
        qry_reps = self.encode_input(qry) if qry else None  # 形状: [bsz_per_device, hidden_dim]
        tgt_reps = self.encode_input(tgt) if tgt else None  # 形状: [bsz_per_device, hidden_dim]

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}  # 仅返回特征（无损失计算）

        # 分布式训练时收集所有进程的特征
        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)  # 收集所有查询特征
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)  # 收集所有目标特征
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        # 计算相似性矩阵（形状: [total_bsz, total_bsz]）
        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)  # 展平为 [batch, batch]

        # 构造目标标签（对角线为正样本）
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))  # 处理多对多匹配

        # 计算交叉熵损失
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size  # 分布式环境下缩放损失（与梯度平均兼容）
        
        return loss
    def _dist_gather_tensor(self, t: Tensor):
        """
        在分布式训练中收集所有进程的张量数据。
        
        参数:
            t (Tensor): 当前进程的特征张量（形状 [bsz_per_device, hidden_dim]）
        
        返回:
            Tensor: 收集后的全局张量（形状 [total_bsz, hidden_dim]）
        """
        t = t.contiguous()  # 确保张量内存连续（避免通信错误）
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]  # 为每个进程创建空张量
        dist.all_gather(all_tensors, t)  # 收集所有进程的张量到all_tensors列表
        all_tensors[self.process_rank] = t  # 手动将当前进程的张量放回原位（避免重复）
        all_tensors = torch.cat(all_tensors, dim=0)  # 沿批次维度拼接
        return all_tensors
    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor):
        """
        计算查询特征与目标特征的余弦相似性。
        
        参数:
            q_reps (Tensor): 查询特征（形状 [batch_q, hidden_dim]）
            p_reps (Tensor): 目标特征（形状 [batch_p, hidden_dim]）
        
        返回:
            Tensor: 相似性矩阵（形状 [batch_q, batch_p]）
        """
        return torch.matmul(q_reps, p_reps.transpose(0, 1))  # 矩阵乘法实现余弦相似性（假设已归一化）
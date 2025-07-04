import torch
from transformers import Qwen2VLModel
from transformers.cache_utils import Cache,DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from models import LatentAttentionBlock

class BiQwen2VLModel(Qwen2VLModel):  # 直接继承原模型类
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.use_bi_atten = True  # 是否使用双向注意力        
    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask with Phi3->Qwen2VL
    # 这个函数仅修改 _prepare_4d_causal_attention_mask_with_cache_position 为 _prepare_bidirectional_attention_mask
    # 函数支持单双向注意力，可以添加到配置文件
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2VL. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        if self.use_bi_atten:
            causal_mask = self._prepare_bidirectional_attention_mask(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=dtype,
                device=device,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
                config=self.config,
                past_key_values=past_key_values,
            )
        else:
            causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=dtype,
                device=device,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
                config=self.config,
                past_key_values=past_key_values,
            )
            

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_bidirectional_attention_mask(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2VLConfig,
        past_key_values: Cache,
    ):
        min_dtype = torch.finfo(dtype).min
        if attention_mask is not None and attention_mask.dim() == 4:
            bidirectional_mask = attention_mask
        else:
            # 创建全连接的基础掩码
            bidirectional_mask = torch.zeros((batch_size, 1, sequence_length, target_length), dtype=dtype, device=device)
    
            # 处理 padding 掩码
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    padding_mask = attention_mask[:, None, None, :].to(device)
                    padding_mask = (1.0 - padding_mask) * min_dtype
                    # 创建一个与 bidirectional_mask 形状相同的全零张量
                    full_padding_mask = torch.zeros_like(bidirectional_mask)
                    # 将 padding_mask 的值复制到 full_padding_mask 的相应位置
                    full_padding_mask[:, :, :, :padding_mask.shape[-1]] = padding_mask
                    bidirectional_mask = bidirectional_mask + full_padding_mask
    
        return bidirectional_mask
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2VLConfig,
        past_key_values: Cache,
    ):
        """构建4D因果注意力掩码（支持缓存位置和滑动窗口）
        
        难点：
        - 处理不同维度的输入mask（2D转4D）
        - 滑动窗口机制的掩码生成
        - 缓存位置对齐
        
        参数：
        - attention_mask: 原始注意力掩码（可能为2D或4D）
        - sequence_length: 当前输入序列长度
        - target_length: 目标长度（考虑缓存后的总长度）
        - cache_position: 当前token在序列中的位置信息
        """
    
        # 如果已经是4D掩码直接返回（例如来自前一层的处理）
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
    
        min_dtype = torch.finfo(dtype).min
        # 初始化全mask矩阵（sequence_length x target_length）
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
    
        # 核心逻辑1：生成基础对角线掩码
        # diagonal_attend_mask形状: [batch_size, target_length]
        diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        
        # 核心逻辑2：滑动窗口机制处理
        if config.sliding_window is not None:
            # 确保当前配置支持滑动窗口缓存
            if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                # 生成滑动窗口掩码（限制注意力范围）
                sliding_attend_mask = torch.arange(target_length, device=device) <= (
                    cache_position.reshape(-1, 1) - config.sliding_window
                )
                diagonal_attend_mask.bitwise_or_(sliding_attend_mask)  # 组合两种掩码
    
        # 应用基础掩码
        causal_mask *= diagonal_attend_mask
        
        # 扩展为4D格式：[batch_size, num_heads, seq_len, target_length]
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    
        # 组合原始attention_mask（处理padding等情况）
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # 创建可写副本
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]  # 截断过长的mask
            
            # 将padding信息整合到因果掩码中
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                causal_mask.device
            )
            padding_mask = padding_mask == 0  # 找出需要mask的位置
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    
        return causal_mask

class BiQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = BiQwen2VLModel(config)
        # self.latent_dim_scale = 1
        # hidden_dim = self.config.hidden_dim
        # latent_dim,hidden_dim = int(hidden_dim*latent_dim_scale),hidden_dim
        # self.latent_atten_block = LatentAttentionBlock(latent_dim,hidden_dim,num_heads=8)
        # batch_size,latent_seq_len = hidden_states.size(0),hidden_states.size(1)
        # self.latent_array = torch.randn(batch_size, latent_seq_len, latent_dim).to(device)
        
    def _global_mean_pool(self,hidden_states, labels):
        """
        全局均值池化（所有有效token取平均）

        Args:
            hidden_states: 模型最后一层隐藏状态 [batch_size, seq_len, hidden_dim]
            labels: 每个token的标签 [batch_size, seq_len]（-100表示无效）

        Returns:
            pooled_features: 池化后的特征 [batch_size, hidden_dim]
        """
        # 创建有效掩码（排除-100），并转换为与hidden_states相同的数据类型
        valid_mask = (labels != -100).unsqueeze(-1)  # [batch_size, seq_len, 1]
        valid_mask = valid_mask.to(hidden_states.dtype)  # 确保数据类型一致

        # 计算加权和（自动广播 valid_mask 到 hidden_dim 维度）
        sum_hidden = torch.sum(hidden_states * valid_mask, dim=1)  # [batch_size, hidden_dim]

        # 计算有效token数量（转换为浮点型）
        num_valid = torch.sum(valid_mask, dim=1)  # [batch_size, 1]
        num_valid = torch.clamp(num_valid, min=1e-7)  # 防止除零

        # 均值池化（广播除法）
        pooled_features = sum_hidden / num_valid
        return pooled_features
    # def _latent_atten_pool(self,latent_dim_scale, hidden_states,device): # 传入当前 hidden_states 所在的 device
    #     embed_features = self.latent_atten_block(latent_array,hidden_states).to(device)

# class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
#     _tied_weights_keys = ["lm_head.weight"]
#     def __init__(self, config):
#         super().__init__(config)
#         self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
#         self.model = Qwen2VLModel(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         self.rope_deltas = None  # cache rope_deltas here
#         # Initialize weights and apply final processing
#         self.post_init()



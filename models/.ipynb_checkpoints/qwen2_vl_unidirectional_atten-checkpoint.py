def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    """更新因果注意力掩码，处理不同注意力实现和缓存机制
    
    重点处理三种场景：
    1. Flash Attention 2实现：需要严格的padding对齐
    2. SDPA实现：利用其内置的因果掩码优化
    3. 常规实现：手动构建4D因果掩码
    
    参数：
    - attention_mask: 2D或4D的注意力掩码
    - input_tensor: 输入张量用于获取设备/类型信息
    - cache_position: 缓存位置指示器
    - past_key_values: 缓存机制（支持静态/滑动窗口/动态缓存）
    - output_attentions: 是否输出注意力权重
    """

    # Flash Attention 2的特殊处理逻辑
    if self.config._attn_implementation == "flash_attention_2":
        # 重点：检查右侧padding的情况（Flash Attention严格要求左padding）
        if attention_mask is not None and past_key_values is not None:
            # 计算最后一个位置是否有padding（batch中存在不完整序列）
            is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2VL. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        # 如果存在全1的attention_mask则直接返回None（Flash Attention会自动处理）
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # 处理SDPA实现（利用其内置的is_causal优化）
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)
    using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

    # 难点：SDPA的优化路径判断
    if (
        self.config._attn_implementation == "sdpa"
        and not (using_static_cache or using_sliding_window_cache)  # 静态/滑动窗口缓存需要特殊处理
        and not output_attentions  # 输出注意力权重时需走常规路径
    ):
        # 判断是否可以使用SDPA内置的因果掩码
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            sliding_window=self.config.sliding_window,
            is_training=self.training,
        ):
            return None

    # 常规实现：手动构建4D因果掩码
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min  # 防止数值溢出的最小值
    
    # 难点：确定目标长度（不同缓存机制处理方式不同）
    sequence_length = input_tensor.shape[1]
    if using_sliding_window_cache or using_static_cache:  # 滑动窗口/静态缓存
        target_length = past_key_values.get_max_cache_shape()  # 使用缓存定义的最大长度
    else:  # 动态缓存或无缓存
        target_length = (
            attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1  # 默认计算方式
        )

    # 生成4D因果掩码的核心方法
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

    # 处理SDPA的特殊填充情况
    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type in ["cuda", "xpu"]  # 仅在GPU/XPU上优化
        and not output_attentions
    ):
        # 重点：处理全masked行（确保内存高效注意力路径正常工作）
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask

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
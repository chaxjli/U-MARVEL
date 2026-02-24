from typing import Optional

import torch

# from ..utils import is_torch_npu_available, is_torch_xpu_available, logging
# from ..utils.import_utils import is_torch_greater_or_equal
from transformers.utils import is_torch_npu_available, is_torch_xpu_available, logging
from transformers.utils.import_utils import is_torch_greater_or_equal

logger = logging.get_logger(__name__)


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    # 2.xpu
    #   - torch version >= 2.8
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    # 3.npu
    #   - npu is not supported gqa currently
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8 and not isinstance(key, torch.fx.Proxy)
    if _is_torch_npu_available:
        return False
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None and not isinstance(key, torch.fx.Proxy)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}
    
    # # 到这里的时候 attention_mask 已经被处理成正确的 4 维形状了
    # print("传进来 attention_mask:", None if attention_mask is None else attention_mask.shape)
    # print("传进来 attention_mask:", attention_mask)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    # print("传进来 is_causal:", is_causal)
    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        print("Tracing and is_causal is a tensor, converting to bool")
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)
    
    # print("module:", module, module.is_causal if hasattr(module, "is_causal") else "N/A")
    # print("attention_mask:", None if attention_mask is None else attention_mask.shape)
    # print("^" * 100)
    # print("is_causal:", is_causal)
    # print("sdpa_kwargs:", sdpa_kwargs)
    # print("attention_mask:", attention_mask)
    # print("^" * 100)
    # # 把结果保存下来
    # with open("attention_mask_debug.txt", "w") as f:
    #     f.write("attention_mask: ")
    #     f.write("\n")
    #     f.write(str(attention_mask))
    #     f.write("\n")

    # print("#" * 100)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    # with open("attn_output2.txt", "a") as f:
    #     f.write("attn_output: ")
    #     f.write("\n")
    #     f.write(str(attn_output))
    #     f.write("\n")

    return attn_output, None


def sdpa_attention_forward_v2(
    module: torch.nn.Module,  # 注意力层的模块
    query: torch.Tensor,  # 查询向量（Q）
    key: torch.Tensor,  # 键向量（K）
    value: torch.Tensor,  # 值向量（V）
    attention_mask: Optional[torch.Tensor],  # 注意力掩码，用于避免关注无效位置
    dropout: float = 0.0,  # dropout 比例，默认为 0.0
    scaling: Optional[float] = None,  # 缩放因子，用于缩放点积，默认为 None
    is_causal: Optional[bool] = None,  # 是否启用因果注意力
    **kwargs,  # 其他额外参数
) -> tuple[torch.Tensor, None]:  # 返回一个元组，包含注意力输出和 None
    # 如果输出注意力矩阵（output_attentions=True）或头部掩码（head_mask）存在，发出警告
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    sdpa_kwargs = {}  # 初始化存储 SDPA 特定参数的字典

    # 如果模块具有 `num_key_value_groups` 属性，处理多组键值
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):  # 如果不使用 GQA（Group Query Attention）
            key = repeat_kv(key, module.num_key_value_groups)  # 重复键
            value = repeat_kv(value, module.num_key_value_groups)  # 重复值
        else:
            sdpa_kwargs = {"enable_gqa": True}  # 启用 GQA

    # 如果有注意力掩码，且其维度为 4，修正掩码的形状
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]  # 修正为与键的最后维度匹配

    # 选择是否使用因果（causal）注意力的逻辑。注意，`is_causal` 会根据输入的查询（query）形状动态判断
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # 在 JIT 编译时，如果 `is_causal` 是一个张量，需要将其转换为布尔值
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # 如果在 Ascend NPU 上运行，且掩码不是布尔类型，将其转换为布尔类型
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # 将注意力掩码转换为布尔类型，以确保 SDPA 使用 FlashAttentionScore 优化
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)
    
    # print("query:", query.shape)
    # print("key:", key.shape)
    # print("value:", value.shape)
    # print("attention_mask:", None if attention_mask is None else attention_mask.shape)

    # 使用 `torch.nn.functional.scaled_dot_product_attention` 计算带有 SDPA 优化的缩放点积注意力
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,  # 使用掩码来屏蔽不关注的部分
        dropout_p=dropout,         # 应用 dropout
        scale=scaling,             # 缩放因子
        is_causal=is_causal,       # 是否启用因果注意力
        **sdpa_kwargs,             # 将 SDPA 特定参数传递给函数
    )

    # 将计算得到的注意力输出进行转置和连续化，以符合预期的输出格式
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None  # 返回注意力输出和 None




def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    # 获取维度：B=批次，H=头数，L=query序列长，S=key序列长，E=头维度
    B, H, L, E = query.shape
    S = key.size(-2)
    scale_factor = 1 / math.sqrt(E) if scale is None else scale
    
    # 初始化 attn_bias 为 (B, H, L, S)，而非 (L, S)，兼容批次和头维度
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    # print("初始化 attn_bias",attn_bias)
    # print("#"*100)
    
    if is_causal:
        # 因果掩码形状调整为 (1, 1, L, S)，通过广播适配所有批次和头
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(~temp_mask.unsqueeze(0).unsqueeze(0), float("-inf"))  # 增加B和H维度
    # print("因果掩码 attn_bias",attn_bias)
    # print("#"*100)
    
    if attn_mask is not None:
        # 确保attn_mask能广播到 (B, H, L, S)
        # 例如：若输入是 (B, 1, 1, S)，会自动广播为 (B, H, L, S)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(~attn_mask, float("-inf"))  # 直接使用广播
        else:
            attn_bias += attn_mask  # 浮点掩码同样广播
    # print("应用 attn_mask",attn_bias)
    
    # print("#"*100)
    # 剩余GQA、注意力计算逻辑不变...
    if enable_gqa:
        repeat_factor = query.size(-3) // key.size(-3)
        key = key.repeat_interleave(repeat_factor, dim=-3)
        value = value.repeat_interleave(repeat_factor, dim=-3)
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor  # 形状 (B, H, L, S)
    # print('attn_weight',attn_weight)
    # print("#"*100)
    
    attn_weight += attn_bias
    # print('attn_weight 应用 attn_bias',attn_weight)
    # print("#"*100)
    
    
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # print('attn_weight 应用 softmax',attn_weight)
    # print("#"*100)
    
    
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # print('attn_weight 应用 dropout',attn_weight)
    # print("#"*100)
    
    return attn_weight @ value


def make_key_padding_mask_for_sdpa(input_ids, pad_token_id):
    """
    输入: input_ids (B, L)
    输出: attn_mask (B, 1, 1, L)，适配 SDPA
    """
    key_padding_mask = (input_ids != pad_token_id)  # (B, L)
    return key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)

# import torch.nn.functional as F
# # 使用
# input_ids = torch.tensor([[101, 102, 0, 0],
#                           [101, 200, 300, 0]])
# attn_mask = make_key_padding_mask_for_sdpa(input_ids, pad_token_id=0)

# output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)



def create_biattention_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """
    将 padding mask 转换为 attention mask，用于 Transformer 的自注意力机制。

    参数:
        padding_mask (torch.Tensor): 
            整数张量，形状为 (B, L)，其中 1 表示有效 token，0 表示 padding。

    返回:
        torch.Tensor: 
            布尔张量，形状为 (B, 1, L, L)，其中 attention_mask[b, 0, i, j] 为 True
            当且仅当第 b 个样本的第 j 个位置是有效 token（即未被 padding）。
    """
    # 获取设备信息
    device = padding_mask.device
    
    # 将 padding_mask 转换为布尔类型
    attention_mask = (padding_mask != 0).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
    
    # 扩展张量至 (B, 1, L, L)
    attention_mask = attention_mask.expand(-1, 1, padding_mask.size(1), -1)  # (B, 1, L, L)
    
    # 确保输出张量在相同设备上
    return attention_mask.to(device)
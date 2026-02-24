import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
# p 代表目标概率分布，q 代表输入概率分布(即是模型输出的 logits,我们预测的概率分布)
def compute_kl_divergence_temp(p, q):
    """
    计算 KL 散度
    :param p: 目标概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param q: 输入概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :return: KL 散度值
    # KL 散度的计算公式为 D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    # 其中 P 和 Q 是两个概率分布
    """
    p = F.softmax(p, dim=-1) # 确保 p 是概率分布
    q = F.softmax(q, dim=-1) # 确保 q 是概率分布
    divergence = torch.sum(p * torch.log(p / (q + 1e-8)), dim=-1)
    assert divergence.shape == p.size(0), f"Expected divergence shape {p.size(0)}, but got {divergence.shape}"
    return divergence.mean()

def compute_kl_divergence(p, q):
    """
    计算 KL 散度
    :param p: 目标概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param q: 输入概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :return: KL 散度值
    # KL 散度的计算公式为 D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    # 其中 P 和 Q 是两个概率分布
    """
    p = F.softmax(p, dim=-1) # 确保 p 是概率分布
    q = F.softmax(q, dim=-1) # 确保 q 是概率分布
    divergence = F.kl_div(torch.log(q + 1e-8), p, reduction='batchmean')
    return divergence

def compute_js_divergence(p, q):
    """
    计算两个概率分布之间的 JS 散度
    :param p: 概率分布 P，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param q: 概率分布 Q，形状为 (batch_size, num_classes) 或 (num_classes,)
    :return: JS 散度值
    # JS 散度的计算公式为 D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    # 其中 M = 0.5 * (P + Q)
    """
    # 确保概率分布是有效的，即概率和为 1
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * (p + q)
    divergence = 0.5 * F.kl_div(torch.log(p + 1e-8), m, reduction='batchmean') + \
                 0.5 * F.kl_div(torch.log(q + 1e-8), m, reduction='batchmean')    
    return divergence

def compute_f_divergence(p, q, f_func):
    """
    计算 f - 散度
    :param p: 概率分布 P，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param q: 概率分布 Q，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param f_func: 满足 f - 散度要求的凸函数，函数输入为张量，输出为张量
    :return: f - 散度值
    # f - 散度的计算公式为 D_f(P || Q) = E_Q[f(P / Q)]
    # 其中 E_Q 表示对 Q 的期望
    # 注意：这里的 p 和 q 应该是经过 softmax 处理的概率分布
    # 你可以根据需要替换为其他满足 f - 散度要求的函数，例如 f(x) = x^2 或 f(x) = x^α
    # 这里的 f_func 应该是一个函数，接受一个张量作为输入，并返回一个张量作为输出
    """
    # 确保概率分布是有效的，即概率和为 1
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    ratio = p / (q + 1e-8)  # 防止除零
    divergence = torch.sum(q * f_func(ratio),dim = -1)
    assert divergence.shape == p.size(0), f"Expected divergence shape {p.size(0)}, but got {divergence.shape}"
    return divergence.mean()

def tv_f_function(t):
    return 0.5 * torch.abs(t - 1)

def kl_f_function(t):
    return t * torch.log(t + 1e-8)


def compute_renyi_divergence(p, q, alpha=2):
    """
    计算广义 KL 散度（这里以 Rényi 散度为例）
    :param p: 目标概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param q: 输入概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param alpha: Rényi 散度的阶数，默认为 2
    :return: Rényi 散度值
    #  Rényi 散度的计算公式为 D_α(P || Q) = (1 / (α - 1)) * log(Σ P(x)^α * Q(x)^(1 - α))
    #  其中 P 和 Q 是两个概率分布，α 是 Rényi 散度的阶数
    #  当 α = 1 时，Rényi 散度退化为 KL 散度
    #  当 α = 2 时，Rényi 散度退化为平方的 KL 散度
    #  注意：这里的 p 和 q 应该是经过 softmax 处理的概率分布
    """
    p = F.softmax(p, dim=-1) # 确保 p 是概率分布
    q = F.softmax(q, dim=-1) # 确保 q 是概率分布
    if alpha == 1:
        # 当 alpha 趋近 1 时，退化为标准 KL 散度
        return F.kl_div(torch.log(q + 1e-8), p, reduction='batchmean')
    else:
        divergence = (1 / (alpha - 1)) * torch.log(torch.sum(torch.pow(p, alpha) * torch.pow(q, 1 - alpha), dim=-1))
        assert divergence.shape == p.size(0), f"Expected divergence shape {p.size(0)}, but got {divergence.shape}"
        return divergence.mean()


def compute_generalized_kl_divergence(p, q):
    """
    计算广义 KL 散度（针对完全非归一化的非负向量情况）
    :param p: 非负向量 P，形状为 (batch_size, num_elements) 或 (num_elements,)
    :param q: 非负向量 Q，形状需与 p 一致
    :return: 广义 KL 散度值
    # 广义 KL 散度的计算公式为  D_{gKL}(P|Q) = Σ (P(x) * log(P(x) / Q(x)) + Q(x) - P(x))
    # 其中 P 和 Q 是两个非负向量
    """
    p = p + 1e-8
    q = q + 1e-8
    divergence = torch.sum((p * torch.log(p / q) + q - p), dim=-1)
    assert divergence.shape == p.size(0), f"Expected divergence shape {p.size(0)}, but got {divergence.shape}"
    return divergence.mean()



def compute_hellinger_distance(p, q):
    """
    计算 Hellinger 距离
    :param p: 概率分布 P，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param q: 概率分布 Q，形状为 (batch_size, num_classes) 或 (num_classes,)
    :return: Hellinger 距离值
    # Hellinger 距离的计算公式为 H(P, Q) = 1 / sqrt(2) * || sqrt(P) - sqrt(Q) ||_2
    # 其中 || . ||_2 表示 L2 范数
    """
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    divergence = torch.sqrt(torch.sum(torch.pow(torch.sqrt(p) - torch.sqrt(q), 2), dim=-1))
    assert divergence.shape == p.size(0), f"Expected divergence shape {p.size(0)}, but got {divergence.shape}"
    return divergence.mean()

def compute_bhattacharyya_distance(p, q):
    """
    计算 Bhattacharyya 距离
    :param p: 概率分布 P，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param q: 概率分布 Q，形状需与 p 一致
    :return: Bhattacharyya 距离值
    # Bhattacharyya 距离的计算公式为 D_B(P, Q) = -log(Σ sqrt(P(x) * Q(x)))
    # 其中 P 和 Q 是两个概率分布
    """
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    divergence = -torch.log(torch.sum(torch.sqrt(p * q), dim=-1))
    assert divergence.shape == p.size(0), f"Expected divergence shape {p.size(0)}, but got {divergence.shape}"
    return divergence.mean()


def compute_mse_loss(p, q):
    """
    计算均方误差损失
    :param p: 目标概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param q: 输入概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :return: 均方误差损失值
    # 均方误差损失的计算公式为 MSE(P, Q) = 1 / n * Σ (P(x) - Q(x))^2
    # 其中 n 是样本数量
    """
    loss = F.mse_loss(p, q, reduction='none')
    loss = loss.sum(dim=-1)
    assert loss.shape == p.size(0), f"Expected loss shape {p.size(0)}, but got {loss.shape}"
    return loss.mean()

def compute_ranking_loss(p, q, gamma=0.1):
    """
    :param p: 目标概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param q: 输入概率分布，形状为 (batch_size, num_classes) 或 (num_classes,)
    :param gamma: 边际超参数
    :return: 排序损失值
    # 排序损失的计算公式为 L = Σ max(0, γ - sign(p_i - p_j) * (q_i - q_j))
    # 其中 p_i 和 p_j 是目标概率分布的元素，q_i 和 q_j 是输入概率分布的元素
    # sign 函数返回 1 或 -1，表示 p_i 和 p_j 的大小关系
    # γ 是一个超参数，用于控制损失的边际
    # 该损失函数用于训练排序模型，鼓励模型在预测时保持相对顺序
    # 例如，如果 p_i > p_j，则希望 q_i > q_j
    # 该损失函数的目标是最小化排序错误
    """
    # 确保输入为二维张量
    if p.dim() == 1:
        p = p.unsqueeze(0)
    if q.dim() == 1:
        q = q.unsqueeze(0)

    batch_size, num_classes = p.shape

    # 计算所有文档对的真实相关性得分差值和模型预测得分差值
    p_diff = p.unsqueeze(2) - p.unsqueeze(1)  # [B,N,N]
    q_diff = q.unsqueeze(2) - q.unsqueeze(1)  # [B,N,N]

    # 计算符号函数值,即 sign(p_i - p_j),[B,N,N]
    p_sign = torch.sign(p_diff).to(device=p.device) 

    # 计算损失项, 即 max(0, γ - sign(p_i - p_j) * (q_i - q_j)),[B,N,N]
    loss_terms = torch.max(torch.tensor(0.0, device=p.device), gamma - p_sign * q_diff)

    # 排除对角线元素（自身与自身的比较）,[B,N,N]
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=p.device)
    loss_terms = loss_terms[:, mask]
    # 计算每个样本的损失，[B,N*(N-1)]
    # print(f"loss_terms shape: {loss_terms.shape}, batch_size: {batch_size}, num_classes: {num_classes}")
    # 求和得到最终损失
    loss = loss_terms.sum()/batch_size/ (num_classes * (num_classes - 1))
    return loss
    
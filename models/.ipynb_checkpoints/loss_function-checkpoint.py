import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义各向同性损失
def isotropy_loss(embeddings: torch.Tensor, 
                  lambda_iso: float = 5e1, 
                  normalize: bool = True) -> torch.Tensor:
    """
    改进后的各向同性损失
    Args:
        embeddings: (n, d) 的嵌入矩阵
        lambda_iso: 损失权重
        normalize: 开启维度标准化（关键改进）
    Returns:
        各向同性损失值
    """
    n, d = embeddings.size()
    if n <= 1:
        return torch.tensor(0.0, device=embeddings.device)
    
    # 均值中心化
    embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    
    # 计算协方差矩阵
    cov = cov = torch.cov(embeddings_centered.T, correction=1) + torch.eye(d, device=embeddings.device) * 1e-6
    
    # 计算标准化损失项
    diff = cov - torch.eye(d, device=embeddings.device)
    loss_iso = diff.pow(2).sum()
    
    # 关键改进：维度标准化
    if normalize:
        # 仅计算协方差矩阵的上三角（含对角线）元素数量
        num_elements = d * (d + 1) // 2  # 独立参数数量
        loss_iso /= num_elements  # 使损失值与维度无关
    
    return lambda_iso * loss_iso


class AlphaWeightedCELoss(nn.Module):
    def __init__(self, alpha=1.0,tau=0.1, eps=1e-8):
        super().__init__()
        self.alpha = alpha  # 确保 alpha>0

    def forward(self, cos_sim, labels):
        """
        cos_sim: (b, n) 相似度矩阵
        labels: (b,) 正样本位置
        """
        device = cos_sim.device
        # 提取正样本分数 (b,)
        pos_scores = cos_sim.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # 计算总指数和 (b,)
        sum_exp = torch.logsumexp(cos_sim, dim=1).exp()  # 稳定计算
        
        # 计算正样本指数调整项 (b,)
        adjusted_pos = pos_scores + torch.log(torch.tensor(self.alpha, device=cos_sim.device))
        
        # 分母构造 (alpha*e^pos + sum_neg)
        denominator = adjusted_pos.exp() + (sum_exp - pos_scores.exp())
        
        # 最终损失计算 (b,)
        loss = (denominator.log() - adjusted_pos).mean().item()
        
        return loss    
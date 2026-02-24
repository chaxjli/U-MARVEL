import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
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
    def __init__(self, alpha=1.0,tau=0.05, eps=1e-8):
        super().__init__()
        self.alpha = alpha  # 确保 alpha > 0
        self.tau = tau
        self.eps = eps
        assert self.alpha > 0, "Alpha must be greater than 0"
        assert self.tau > 0, "Tau must be greater than 0"
        assert self.eps > 0, "Epsilon must be greater than 0"
    def forward(self, cos_sim, labels):
        """
        Args:
            cos_sim: (b, n) 相似度矩阵, n 是 batch_size 的倍数；
            labels: (b,) 正样本位置, b 是 batch_size；
        Returns:
            loss: (1,) 损失值；
        remarks: 传入的 cos_sim 相似度矩阵未进行过温度参数的处理；
        """
        device = cos_sim.device
        pos_scores = cos_sim.gather(1, labels.unsqueeze(1)).squeeze(1)                 # 提取正样本分数 (b,)
        sum_exp = torch.logsumexp(cos_sim, dim=1).exp()                                # 计算总指数和 (b,), 稳定计算
        adjusted_pos = pos_scores + torch.log(torch.tensor(self.alpha, device=device)) # 计算正样本指数调整项 (b,)
        denominator = adjusted_pos.exp() + (sum_exp - pos_scores.exp()) + self.eps     # 分母构造 (alpha*e^pos + sum_neg)
        loss = (denominator.log() - adjusted_pos).mean().item()                        # 最终损失计算 (b,)
        
        return loss

class FusedLoss(nn.Module):
    def __init__(self, alpha=1.0, tau=0.07, eps=1e-8):
        super().__init__()
        self.alpha = alpha  # 正样本项的加权系数
        self.tau = tau      # 温度参数
        self.eps = eps      # 数值稳定性常数
        # 确保参数合法性
        assert self.alpha > 0, "Alpha must be greater than 0"
        assert self.tau > 0, "Tau must be greater than 0"
        assert self.eps >= 0, "Epsilon must be greater than 0"

    def forward(self, cos_sim, labels):
        """
        Args:
            cos_sim: (b, n) 输入的相似度矩阵，未经过温度缩放，但是已经减去了每一行的最大值，保持数值的稳定性
            labels: (b,) 每个样本的正样本位置索引
            remarks: cos_sim 的形状为 (b, n), 其中 b 是 batch_size, n 是负样本数量的倍数，存在显存优化的空间
        Returns:
            loss: 计算得到的损失值，保留梯度信息
        """
        
        device = cos_sim.device
        b, n = cos_sim.shape
        # 提取正样本分数并增强 (b,), 将其存储为 fp32 类型保存精度
        pos_scores = cos_sim.gather(1, labels.unsqueeze(1)).squeeze(1).float()  # 提取正样本分数
        enhanced_pos = self._enhance_pos(pos_scores)  # 正样本增强, φ_m(s_p^i), enhanced_pos 对应的 loss 的分子的指数部分；
        
        # # 监控 enhanced_pos 最大值最小值是否溢出
        # rank0_print(f"pos_scores: {pos_scores.min().item()} {torch.argmin(pos_scores).item()} {pos_scores.max().item()} {torch.argmax(pos_scores).item()}")
        # rank0_print(f"enhanced_pos: {enhanced_pos.min().item()} {enhanced_pos.max().item()}")
        

        # 生成 mask 以排除正样本位置 (b, n), 主要目的是为了避免正样本参与负样本求和的运算
        mask = torch.zeros_like(cos_sim, dtype=torch.bool).to(device)  # 创建与 cos_sim 相同形状的布尔型张量
        mask.scatter_(1, labels.unsqueeze(1), True)                    # 正样本位置标记为True
        
        # 调制负样本分数 (b, n)
        mod_neg_scores = self._modulate_neg(cos_sim.float(),mask)         # 负样本调制, θ_m(s_n^{i,j})
        
        # rank0_print(f"mod_neg_scores: {mod_neg_scores.min().item()} {mod_neg_scores.max().item()}")
        # # 对正负样本进行最大值处理，避免数值溢出
        # max_values_neg = mod_neg_scores.masked_fill(mask, -float('inf')).max(dim=1, keepdim=True).values.detach()  # 负样本最大值计算，分离梯度（避免梯度传播）
        # max_values_pos = enhanced_pos.detach()                                    # 正样本最大值计算，分离梯度（避免梯度传播）
        # max_values = torch.max(max_values_neg, max_values_pos.unsqueeze(1))       # 计算正负样本的最大值
        # mod_neg_scores = mod_neg_scores - max_values                              # 调制之后的负样本减去最大值，避免数值溢出
        # enhanced_pos = enhanced_pos - max_values.squeeze(1)                       # 加强之后的正样本减去最大值，避免数值溢出
        
        # 计算负样本权重, 需要取对数，因为最后会和负样本一块进行 exp 操作：log(ψ_m(s_n^{i,j}))
        weighted_neg = torch.log(self._weight_neg(cos_sim.float(),mask) + self.eps)     # self.eps 防止对 0 取对数，指数溢出
        mod_neg_scores_weight = mod_neg_scores + weighted_neg     # 应用对样本进行加权, θ_m(s_n^{i,j}) + log(ψ_m(s_n^{i,j}))  

        # # 监控 cos_sim,mod_neg_scores，weighted_neg，mod_neg_scores_weight 最大值最小值是否溢出
        # rank0_print("max_values_neg:", max_values_neg.min().item(), max_values_neg.max().item(),max_values_neg.shape)
        # rank0_print("max_values_pos:", max_values_pos.min().item(), max_values_pos.max().item(),max_values_pos.shape)
        # rank0_print(f"max_values: {max_values.min().item()} {max_values.max().item()}", max_values.shape)
        # rank0_print(f"cos_sim: {cos_sim.min().item()} {torch.argmin(cos_sim).item()%480} {cos_sim.max().item()}  {torch.argmax(cos_sim).item()%480}")
        # rank0_print(f"已做最大值处理 mod_neg_scores: {mod_neg_scores.min().item()} {mod_neg_scores.max().item()}")
        # rank0_print(f"已做最大值处理 enhanced_pos: {enhanced_pos.min().item()} {enhanced_pos.max().item()}")
        # rank0_print(f"weighted_neg_log: {weighted_neg.min().item()} {weighted_neg.max().item()}")
        # rank0_print(f"mod_neg_scores_weight: {mod_neg_scores_weight.min().item()} {mod_neg_scores_weight.max().item()}")
        

        # 将正样本位置分数设为 -无穷，避免其参与负样本求和, 计算负样本的指数和 (b,)
        mod_neg_scores_weight_masked = mod_neg_scores_weight.masked_fill(mask, -float('inf'))
        sum_neg_logits = torch.exp(mod_neg_scores_weight_masked) # 计算负样本的指数
        sum_neg_exp = torch.sum(sum_neg_logits, dim=1)           # 计算负样本的指数和 (b,)

        # # 监控 mod_neg_scores_weight_masked, sum_neg_logits, sum_neg_exp 最大值最小值是否溢出
        # rank0_print(f"mod_neg_scores_weight_masked: {mod_neg_scores_weight_masked.min().item()} {mod_neg_scores_weight_masked.max().item()}")
        # rank0_print(f"sum_neg_exp: {sum_neg_logits.min().item()} {sum_neg_logits.max().item()}")
        # rank0_print(f"sum_neg_sum: {sum_neg_exp.min().item()} {sum_neg_exp.max().item()}")
    
        # 正样本加权调整 (b,), 这一部分主要针对 loss 分母当中正样本部分的 指数；
        weighted_pos = torch.log(self._weight_pos(pos_scores)+ self.eps)  # 正样本加权, log(a_m(s_p^i)) 
        adjusted_pos = (enhanced_pos + weighted_pos).exp()

        # # 监控 weighted_pos,adjusted_pos 最大值最小值是否溢出
        # rank0_print(f"weighted_pos_log: {weighted_pos.min().item()} {weighted_pos.max().item()}")
        # rank0_print(f"adjusted_pos: {adjusted_pos.min().item()} {adjusted_pos.max().item()}")
        
        # 构造分母项：负样本加权的指数和 + 正样本加权的指数 + eps， eps 是为了避免数值溢出，形状是 (b,)
        denominator = adjusted_pos + sum_neg_exp + torch.tensor(self.eps, dtype=torch.float32)
        
        # # 监控 denominator 最大值最小值是否溢出
        # rank0_print(f"denominator: {denominator.min().item()} {denominator.max().item()}")
        # rank0_print(f"denominator_log: {torch.log(denominator).min().item()} {torch.log(denominator).max().item()}")
        # rank0_print(f"enhanced_pos: {enhanced_pos.min().item()} {enhanced_pos.max().item()}")


        # 计算最终损失 (log(denominator) - enhanced_pos) 的均值， 形状是 (b,)
        loss = (torch.log(denominator) - enhanced_pos).mean()
        loss = loss.to(cos_sim.dtype)  # 转回原类型
        return loss
    
    # 定义各调制函数
    def _enhance_pos(self, pos_scores):
        """
        正样本增强函数
        Args:
            pos_scores: (b,) 正样本分数
        Returns:
            enhanced_pos: (b,) 增强后的正样本分数
        """
        enhanced_pos = pos_scores.clone().to(pos_scores.device)  # 深拷贝，避免原地操作
        enhanced_pos = enhanced_pos/self.tau
        return enhanced_pos
    
    def _modulate_neg(self, cos_sim, mask=None):
        """
        负样本调制函数
        Args:
            cos_sim: (b, n) 相似度矩阵
        Returns:
            modulated_neg: (b, n) 调制后的负样本分数
        """
        modulated_neg = cos_sim.clone().to(cos_sim.device)  # 深拷贝，避免原地操作
        modulated_neg = modulated_neg/self.tau
        return modulated_neg

    def _weight_pos(self, pos_scores):
        """
        正样本加权函数
        Args:
            pos_scores: (b,) 正样本分数
        Returns:
            weighted_pos: (b,) 加权后的正样本分数
        """
        weighted_pos = torch.full_like(pos_scores, self.alpha).to(pos_scores.device)
        return weighted_pos
    
    def _weight_neg(self, cos_sim, mask=None):
        """
        负样本加权函数
        Args:
            cos_sim: (b, n) 相似度矩阵
        Returns:
            weighted_neg: (b, n) 加权后的负样本分数
        """
        weighted_neg = torch.full_like(cos_sim, self.alpha).to(cos_sim.device)
        return weighted_neg

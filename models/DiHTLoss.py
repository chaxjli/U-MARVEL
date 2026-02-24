from models.loss_function import FusedLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
import numpy as np
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
class DiHTLoss(FusedLoss):
    def __init__(self, alpha=1.0, tau=0.05, eps=1e-8, margin=0.3):
        """
        Args:
            alpha: 权重参数， 这里的是用来对正样本做加权处理的；
            tau: 温度参数，保持使用 M-BEIR 当中的温度参数，DiHT论文当中没有说明
            eps: 数值稳定性参数
            margin: 边界参数， DiHT 用不到这个参数；
        """
        # 调用父类构造函数
        super().__init__(alpha, tau, eps)
        self.alpha = 0.95   # 重置 alpha
        self.beta = 0.1     # 定义 beta
        self.tau = 0.05     # 使用 M-BEIR 的 tau
        rank0_print(f"DiHTLoss: alpha={self.alpha}, tau={self.tau}, eps={self.eps}, margin={margin}")
    
    def _enhance_pos(self, pos_scores):
        """
        正样本增强函数,φ_m(s_p^i) = (s_p^i)/τ
        Args:
            pos_scores: (b,) 正样本分数
        Returns:
            enhanced_pos: (b,) 增强后的正样本分数
        """
        enhanced_pos = pos_scores.clone().to(pos_scores.device)
        enhanced_pos = enhanced_pos/self.tau
        return enhanced_pos
    def _modulate_neg(self, cos_sim,mask=None):
        """
        负样本调制函数,θ_m(s_n^{i,j}) = s_n^{i,j}/τ
        Args:
            cos_sim: (b, n) 负样本分数
        Returns:
            modulated_neg: (b, n) 调制后的负样本分数
        """
        modulated_neg = cos_sim.clone().to(cos_sim.device)
        modulated_neg = modulated_neg/self.tau
        return modulated_neg
    def _weight_pos(self, pos_scores):
        """
        正样本加权函数, a_m(s_p^i) = alpha;
        这里的正样本加权是 alpha
        Args:
            pos_scores: (b,) 正样本分数
        Returns:
            weighted_pos: (b,) 加权后的正样本分数
        """
        weighted_pos = torch.full_like(pos_scores, self.alpha).to(pos_scores.device)
        return weighted_pos
    
    def _weight_neg(self, cos_sim,mask=None):
        """
        负样本加权函数, ψ_m(s_n^{i,j}) = (s[i,j] * (n-1)) / (Σ_{k≠j} s[i,k]);
        Args:
            cos_sim: (b, n) 负样本分数
        Returns:
            weighted_neg: (b, n) 加权后的负样本分数
        """
        b,n = cos_sim.size()
        weighted_neg = cos_sim.clone()
        weighted_neg = (self.beta*weighted_neg).exp()  # 这里的负样本加权是 beta
        weighted_neg = weighted_neg*(n-1) / (torch.sum(weighted_neg, dim=1, keepdim=True) - weighted_neg + self.eps)
        return weighted_neg         
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
class LLaVELoss(FusedLoss):
    def __init__(self, alpha=1.0, tau=0.05, eps=1e-8, margin=0.3):
        """
        Args:
            alpha: 权重参数, 这里的是用来对负样本做加权处理的；
            tau: 温度参数, 保持使用 M-BEIR 当中的温度参数； 
            eps: 数值稳定性参数；
            margin: 边界参数, LLaVE 用不到这个参数；
        """
        # 调用父类构造函数
        super().__init__(alpha, tau, eps)
        self.alpha = 9.0   # 使用 LLaVE 的 alpha
        self.tau = 0.05    # 使用 M-BEIR 的 tau
     
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
        weighted_pos = torch.full_like(pos_scores, 1.0).to(pos_scores.device)
        return weighted_pos
    
    def _weight_neg(self, cos_sim,mask=None):
        """
        负样本加权函数: ψ_m(s_n^{i,j}) = [alpha*(s_n^{i,j})].exp();
        Args:
            cos_sim: (b, n) 负样本分数
        Returns:
            weighted_neg: (b, n) 加权后的负样本分数
        """
        b,n = cos_sim.size()
        weighted_neg = cos_sim.detach()
        weighted_neg = (self.alpha*weighted_neg).exp()
        # rank0_print(cos_sim.requires_grad)  # 输出应为 False
        # rank0_print(weighted_neg.requires_grad)  # 输出应为 False
        return weighted_neg         
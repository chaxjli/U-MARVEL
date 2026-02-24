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

class SoftCSELoss_Weight(FusedLoss):
    def __init__(self, alpha=1.0, tau=0.05, eps=1e-8, margin=0.3):
        """
        Args:
            alpha: 权重参数, SoftCSELoss 用不到这个参数；
            tau: 温度参数, 保持使用 M-BEIR 当中的温度参数； 
            eps: 数值稳定性参数；
            margin: 边界参数, SoftCSELoss 用不到这个参数；
        """
        # 调用父类构造函数
        super().__init__(alpha, tau, eps)
        self.tau = 0.10    # 使用 M-BEIR 的 tau
        rank0_print(f"SoftCSELoss_Weight: tau={self.tau}, eps={eps}")
     
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
        负样本加权函数: ψ_m(s_n^{i,j}) = [1-(s_n^{i,j})]/(1-1/(n-1));
        Args:
            cos_sim: (b, n) 负样本分数
        Returns:
            weighted_neg: (b, n) 加权后的负样本分数
        """
        b,n = cos_sim.size()
        weighted_neg = cos_sim.clone()
        weighted_neg = (1 - weighted_neg)/(1-1/(n-1)+self.eps)
        return weighted_neg 


class SoftCSELoss_Temperature(FusedLoss):
    def __init__(self, alpha=1.0, tau=0.05, eps=1e-8, margin=0.3):
        """
        Args:
            alpha: 权重参数, SoftCSELoss 用不到这个参数；
            tau: 温度参数, 保持使用 M-BEIR 当中的温度参数； 
            eps: 数值稳定性参数；
            margin: 边界参数, SoftCSELoss 用不到这个参数；
        """
        # 调用父类构造函数
        super().__init__(alpha, tau, eps)
        # 使用 M-BEIR 的 tau = 0.05，loss 会接近 0.0，选择调大 tau = 0.1
        # pytorch 封装的 CrossEntropyLoss 会减去 max，为保持一致进行同样的操作，此时可以设置 tau = 0.05；
        # 这里的 tau 代表的是温度参数, 0.05 训练的时候 cos_sim 会突然变成 nan，这是为什么；
        self.tau = 0.10
        rank0_print(f"SoftCSELoss_Temperature: tau={self.tau}, eps={eps}")
     
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
        负样本调制函数,θ_m(s_n^{i,j}) = s_n^{i,j}/τ{i,j},其中 τ{i,j} = (n-2)τ/[(n-1)(1-s_n^{i,j})];
        Args:
            cos_sim: (b, n) 负样本分数, b 代表 query 的数量, n 代表 正样本+负样本 的数量
        Returns:
            modulated_neg: (b, n) 调制后的负样本分数
        """
        modulated_neg = cos_sim.clone().to(cos_sim.device)
        b,n = modulated_neg.size()
        modulated_neg = modulated_neg*(1-modulated_neg)*(n-1)/((n-2)*self.tau)
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
        weighted_neg = cos_sim.clone()
        weighted_neg = torch.full_like(weighted_neg, 1.0).to(cos_sim.device)
        return weighted_neg 
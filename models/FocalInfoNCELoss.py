from models.loss_function import FusedLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
class FocalInfoNCELoss(FusedLoss):
    def __init__(self, alpha=1.0, tau=0.05, eps=1e-8,margin=0.3):
        """
        Args:
            tau: 温度参数
            margin: 边界参数
        """
        super().__init__(alpha, tau, eps)
        # 原始版的 margin 是 0.3, margin 是一个超参数
        self.margin = 0.2
        self.tau = 0.05
        self.eps = 1e-8
        rank0_print(f"FocalInfoNCELoss: margin={self.margin}, tau={self.tau}, eps={self.eps}")

    def _enhance_pos(self, pos_scores):
        """
        正样本增强函数,φ_m(s_p^i) = (s_p^i)^2/τ
        这里的正样本增强是平方操作
        Args:
            pos_scores: (b,) 正样本分数
        Returns:
            enhanced_pos: (b,) 增强后的正样本分数
        """
        enhanced_pos = pos_scores.clone().to(pos_scores.device)
        enhanced_pos = enhanced_pos.pow(2)
        enhanced_pos = enhanced_pos/self.tau
        return enhanced_pos
    
    def _modulate_neg(self, cos_sim,mask=None):
        """
        负样本调制函数,θ_m(s_n^{i,j}) = (s_n^{i,j})(s_n^{i,j}+margin)/τ
        这里的负样本调制是带margin的平方操作
        Args:
            cos_sim: (b, n) 相似度矩阵
        Returns:
            modulated_neg: (b, n) 调制后的负样本分数
        """
        modulated_neg = cos_sim.clone().to(cos_sim.device)
        modulated_neg = modulated_neg*(modulated_neg + self.margin)
        modulated_neg = modulated_neg/self.tau
        return modulated_neg
    
    def _weight_pos(self, pos_scores):
        """
        正样本加权函数, a_m(s_p^i) = 1.0; 这里的正样本加权是 1.0
        Args:
            pos_scores: (b,) 正样本分数
        Returns:
            weighted_pos: (b,) 加权后的正样本分数
        """
        weighted_pos = torch.full_like(pos_scores, 1.0).to(pos_scores.device)
        return weighted_pos
    
    def _weight_neg(self, cos_sim, mask=None):
        """
        负样本加权函数, ψ_m(s_n^{i,j}) = 1.0; 这里的负样本加权是 1.0
        Args:
            cos_sim: (b, n) 相似度矩阵
        Returns:
            weighted_neg: (b, n) 加权后的负样本分数
        """
        weighted_neg = torch.full_like(cos_sim, 1.0).to(cos_sim.device)
        return weighted_neg



class FocalInfoNCEABSLoss(FusedLoss):
    def __init__(self, alpha=1.0, tau=0.05, eps=1e-8,margin=0.3):
        """
        Args:
            tau: 温度参数
            eps: 数值稳定性参数
            margin: 边界参数
        """
        super().__init__(alpha, tau, eps)
        self.margin = 0.3
        self.eps = 1e-8
        self.tau = 0.05
        rank0_print(f"FocalInfoNCELoss: margin={self.margin}, tau={self.tau}, eps={self.eps}")
    
    def _enhance_pos(self, pos_scores):
        """
        正样本增强函数,φ_m(s_p^i) = (s_p^i)(|s_p^i|)/τ
        Args:
            pos_scores: (b,) 正样本分数
        Returns:
            enhanced_pos: (b,) 增强后的正样本分数
        """
        enhanced_pos = pos_scores.clone().to(pos_scores.device)
        enhanced_pos = enhanced_pos*torch.abs(enhanced_pos)
        enhanced_pos = enhanced_pos/self.tau
        return enhanced_pos
    
    def _modulate_neg(self, cos_sim,mask=None):
        """
        负样本调制函数,θ_m(s_n^{i,j}) = (s_n^{i,j})(|s_n^{i,j}|+margin)/τ
        Args:
            cos_sim: (b, n) 相似度矩阵
        Returns:
            modulated_neg: (b, n) 调制后的负样本分数
        """
        modulated_neg = cos_sim.clone().to(cos_sim.device)
        modulated_neg = modulated_neg*(torch.abs(modulated_neg) + self.margin)
        modulated_neg = modulated_neg/self.tau
        return modulated_neg
    
    def _weight_pos(self, pos_scores):
        """
        正样本加权函数, a_m(s_p^i) = 1.0; 这里的正样本加权是 1.0
        Args:
            pos_scores: (b,) 正样本分数
        Returns:
            weighted_pos: (b,) 加权后的正样本分数
        """
        weighted_pos = torch.full_like(pos_scores, 1.0).to(pos_scores.device)
        return weighted_pos
    
    def _weight_neg(self, cos_sim, mask=None):
        """
        负样本加权函数, ψ_m(s_n^{i,j}) = 1.0;
        Args:
            cos_sim: (b, n) 相似度矩阵
        Returns:
            weighted_neg: (b, n) 加权后的负样本分数
        """
        weighted_neg = torch.full_like(cos_sim, 1.0).to(cos_sim.device)
        return weighted_neg
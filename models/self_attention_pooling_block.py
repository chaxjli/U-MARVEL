class SelfAttentionPoolingImproved(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttentionPoolingImproved, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 维度保持: hidden_dim -> hidden_dim
            nn.Tanh(),                          # 非线性激活
            nn.Linear(hidden_dim, 1)            # 最终映射到标量分数: hidden_dim -> 1
        )

    def forward(self, hidden_states, labels):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            labels: [batch_size, seq_len]
        Returns:
            pooled_features: [batch_size, hidden_dim]
        """        
        batch_size, seq_len, hidden_dim = hidden_states.size()

        # hidden_states: [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len]
        attention_scores = self.attention(hidden_states).squeeze(-1)
        
        # mask 形状: [batch_size, seq_len]
        mask = (labels != -100)
        
        # 处理无效位置的注意力分数:[batch_size, seq_len]
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # 计算注意力权重：[batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        
        # 处理全被mask的样本（防止NaN）
        valid_mask = mask.any(dim=1, keepdim=True).unsqueeze(-1)
        # 将valid_mask扩展到[batch_size, seq_len, 1]
        attention_weights = attention_weights * valid_mask.float()
        
        # （全被mask的样本）
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        
        # 加权求和得到池化特征
        # hidden_states: [batch_size, seq_len, hidden_dim]
        # attention_weights: [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        pooled_features = torch.sum(hidden_states * attention_weights, dim=1)
        return pooled_features
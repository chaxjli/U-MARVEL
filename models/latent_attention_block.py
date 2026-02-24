import torch
import torch.nn as nn

# 定义潜在注意力模块
# 不对 out 进行池化操作，是否池化还是选取相应的 <emb> 由调用者决定
class LatentAttentionBlock(nn.Module):
    # 初始化需要传入 latent_dim(潜在注意力模块的维度) hidden_dim(隐藏层维度) num_heads(多头注意力机制的头数)
    def __init__(self,latent_dim, hidden_dim, num_heads=28):
        super(LatentAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # 初始化潜在数组，形状为 (1, 1, latent_dim)
        self.latent_array = nn.Parameter(torch.randn(1, 1, latent_dim)) 
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num_heads"
        self.Wv = nn.Linear(latent_dim, hidden_dim)
        self.Wk = nn.Linear(latent_dim, hidden_dim)
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
    def forward(self,decoder_output):
        """
        Args: 
             latent_array:   [batch_size,decoder_seq_len, latent_dim]
             decoder_output: [batch_size,decoder_seq_len,hidden_dim]
        """
        batch_size,decoder_seq_len, hidden_dim= decoder_output.size()
        latent_array = self.latent_array.expand(batch_size, decoder_seq_len, -1)  # 扩展潜在数组至当前批大小[batch_size, decoder_seq_len, latent_dim]
        latent_seq_len = latent_array.size(1)
        # 生成 Q, K, V
        V = self.Wv(latent_array)    # [batch_size, decoder_seq_len, hidden_dim]
        K = self.Wk(latent_array)    # [batch_size, decoder_seq_len, hidden_dim]
        Q = self.Wq(decoder_output)  # [batch_size, decoder_seq_len, hidden_dim]

        # 拆分为多头 # [batch_size, num_heads, decoder_seq_len, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数 [batch_size, num_heads, decoder_seq_len, decoder_seq_len]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        attn_probs = self.softmax(attn_scores)

        # 计算加权值
        attended_values = torch.matmul(attn_probs, V)  # [batch_size, num_heads, decoder_seq_len, head_dim]

        # 合并多头  [batch_size, decoder_seq_len, hidden_dim]
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  
        output = self.mlp(attended_values)
        return output

import torch
from torch import nn
from .layers import TransformerBlock, SelfAttentionBlock
import einops
from .layers import PositionEmbeddingSine1D
import torch.nn.functional as F

class ConditionalMemoryEncoder(nn.Module):
    def __init__(self, sam_hidden_dim=256):
        super().__init__()
        self.transformerBlock = SelfAttentionBlock(d_model=sam_hidden_dim, nhead=8)
        self.decision_token = nn.Embedding(1, sam_hidden_dim)
        self.decision_class_head = nn.Linear(sam_hidden_dim, 2)

    def forward(self, obj_ptr_zero, obj_ptr_two):
        B = obj_ptr_zero.shape[0]
        obj_ptr_zero = obj_ptr_zero.unsqueeze(0)
        obj_ptr_two = obj_ptr_two.unsqueeze(0)

        input_sequence = torch.cat([self.decision_token.weight.repeat(B,1).unsqueeze(0), obj_ptr_zero, obj_ptr_two], dim=0)
        out = self.transformerBlock(input_sequence)
        out = self.decision_class_head(out[0])
        # return CLS token
        return out

class ResidualTextEncoder(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_tokens):
        global_vector = text_tokens.mean(dim=1)  # [B, D]
        return global_vector

class SpatialAttentionMap(nn.Module):
    def forward(self, predicted_mask):
        A_s = torch.sigmoid(predicted_mask) # 简单的 Sigmoid 激活
        return A_s

class AffineModulation(nn.Module):
    def __init__(self, D_text, D_visual):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D_text, D_text),
            nn.ReLU(),
            nn.Linear(D_text, D_visual * 2) # 输出 gamma 和 beta
        )
        self.D_visual = D_visual

    def forward(self, tau_g):
        gamma_beta = self.mlp(tau_g)
        gamma, beta = torch.split(gamma_beta, self.D_visual, dim=-1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


class CrossAttentionUpdate(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, F_t, prev_memory):
        query = F_t.mean(dim=1).unsqueeze(1) 
        
        new_memory, _ = self.cross_attn(query, prev_memory, prev_memory)
        return self.norm(new_memory.squeeze(1)) # 返回更新后的 Memory 向量


class TextConditionedMemoryEncoder(nn.Module):
    def __init__(self, sam_hidden_dim=256, text_dim=512, visual_dim=256):
        super().__init__()
        self.text_encoder = ResidualTextEncoder(text_dim)
        self.spatial_attn = SpatialAttentionMap()
        self.affine_mod = AffineModulation(text_dim, visual_dim)
        self.cross_attn_update = CrossAttentionUpdate(visual_dim, nhead=8)


        self.aux_transformerBlock = nn.TransformerEncoderLayer(d_model=sam_hidden_dim, nhead=8, dim_feedforward=1024, batch_first=False)
        self.aux_decision_token = nn.Embedding(1, sam_hidden_dim)
        self.aux_decision_class_head = nn.Linear(sam_hidden_dim, 2)
        # -----------------------------------------------------

    def forward(self, text_tokens, predicted_mask, visual_features, prev_memory):

        B, D_visual, H, W = visual_features.shape

        tau_g = self.text_encoder(text_tokens) # [B, D_text]

        A_s = self.spatial_attn(predicted_mask) # [B, 1, H, W]

        F_v_prime = visual_features * (1 + A_s) # 增强 [B, D_visual, H, W]

        gamma, beta = self.affine_mod(tau_g) # gamma/beta: [B, D_visual, 1, 1]
        
        F_t = gamma * F_v_prime + beta # [B, D_visual, H, W]

        F_t_flat = F_t.flatten(2).permute(0, 2, 1)

        # M_t = CrossAttention(Q=F_t, K=M_{t-1}, V=M_{t-1})
        M_t = self.cross_attn_update(F_t_flat, prev_memory) # [B, D_visual]

        return M_t # [B, D_visual]
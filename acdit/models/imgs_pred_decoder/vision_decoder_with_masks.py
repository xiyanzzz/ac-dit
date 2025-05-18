import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.models.vision_transformer import Mlp, RmsNorm, PatchEmbed
from einops import rearrange, repeat


def modulate(x, shift, scale): 
    """
    期待三维张量 (B,T,D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Image Prediction Interval                  #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                          Attention Layers                                     #
#################################################################################

class Attention(nn.Module):
    """
    Masked multi-head attention module modified from timm.models.vision_transformer.py
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.3, # TODO: make it in config
            proj_drop: float = 0.1, # TODO: make it in config
            norm_layer: nn.Module = RmsNorm,
            fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            mask = mask.reshape(B, 1, 1, N)
            mask = mask.expand(-1, -1, N, -1)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    Masked cross-attention layer modified from RDT
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0., # TODO: make it in config
            proj_drop: float = 0., # TODO: make it in config
            norm_layer: nn.Module = RmsNorm,
            fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask = None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x
 

#################################################################################
#                Vision Repersentaion Projection Layers                         #
#################################################################################

class VisionRepresentationProjector(nn.Module):
    """
    Projects vision encoder tokens to decoder hidden dim
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.projector(x)


#################################################################################
#                         Core Vision Transformer Module                        #
#################################################################################

class VisionTransformerBlock(nn.Module):
    """
    DiT block modified with:
    1. adding cross-attention conditioning
    2. using rmsnorm instead of layernorm
    """
    def __init__(self, hidden_size, 
                 num_heads, 
                 mlp_ratio=4.0, 
                 attn_drop=0.3,
                 proj_drop=0.1, 
                 qk_norm=False, 
                 cross_attn_drop=0., 
                 cross_proj_drop=0., 
                 cross_qk_norm=True, 
                 mlp_drop=0., 
                 **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, 
                              qkv_bias=True, attn_drop=attn_drop, 
                              proj_drop=proj_drop, qk_norm=qk_norm, 
                              **block_kwargs)
        
        self.norm2 = RmsNorm(hidden_size, affine=True, eps=1e-6)
        # add cross-attention conditioning with rmsNorm
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, 
                                         qkv_bias=True, qk_norm=cross_qk_norm, 
                                         attn_drop=cross_attn_drop, proj_drop=cross_proj_drop, 
                                         **block_kwargs)
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, 
                       drop=mlp_drop) # TODO: make it in config
        

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 7 * hidden_size, bias=True)
        )

    def forward(self, x, c, g, self_mask=None, cross_mask=None):
        shift_msa, scale_msa, gate_msa, gate_cross, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(7, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), self_mask)
        x = x + gate_cross.unsqueeze(1) * self.cross_attn(self.norm2(x), g, cross_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


#################################################################################
#                         Vision Decoder with Mask Tokens                        #
#################################################################################

class FinalLayer(nn.Module):
    """
    Final layer of the Vision Decoder, similar to DiT's final layer.
    """
    def __init__(self, hidden_size, patch_resolution):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, patch_resolution)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class VisionDecoderWithMasks(nn.Module):
    """
    Vision Decoder with Mask Tokens for future image prediction.
    Uses masked image modeling to reconstruct future images based on vision encoder tokens.
    """
    def __init__(
        self,
        input_size=144,              # Input image size
        patch_size=16,               # Image patch size
        in_channels=3,               # Input channels
        hidden_size=256,             # Hidden feature dimension
        encoder_hidden_size=512,     # Vision encoder output dimension
        depth=6,                     # Number of transformer layers
        num_heads=8,                 # Number of attention heads
        mlp_ratio=4.0,               # MLP ratio
        num_cameras=2,               # Number of cameras
        mask_ratio=0.75,             # Ratio of patches to mask
        max_future_step=10,          # Maximum future time step
        attn_drop=0.3,               # Attention dropout
        proj_drop=0.1,               # Projection dropout
        qk_norm=False,               # Query-key normalization
        cross_attn_drop=0.,          # Cross-attention dropout
        cross_proj_drop=0.,          # Cross-projection dropout
        cross_qk_norm=True,          # Cross-query-key normalization
        mlp_drop=0.,                 # MLP dropout
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_cameras = num_cameras
        self.mask_ratio = mask_ratio
        self.encoder_hidden_size = encoder_hidden_size
        
        # Image patch embedding
        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_patches = self.patch_embed.num_patches
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Position embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        
        # Camera position embedding
        self.camera_embed = nn.Parameter(torch.zeros(1, num_cameras, hidden_size), requires_grad=False)
        
        # Future timestep embedder
        self.future_timestep_embedder = TimestepEmbedder(hidden_size)
        
        # Vision token projector - projects encoder tokens to decoder dimension
        self.vision_projector = VisionRepresentationProjector(encoder_hidden_size, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                                    proj_drop=proj_drop, qk_norm=qk_norm, cross_attn_drop=cross_attn_drop, 
                                    cross_proj_drop=cross_proj_drop, cross_qk_norm=cross_qk_norm, 
                                    mlp_drop=mlp_drop)
            for _ in range(depth)
        ])
        
        # Final layer for patch reconstruction
        self.final_layer = FinalLayer(hidden_size, (patch_size ** 2) * in_channels)
        
        self.max_future_step = max_future_step
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)
        
        # Initialize mask tokens
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize position embeddings with sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize camera position embedding
        camera_embed = get_1d_sincos_pos_embed_from_grid(self.camera_embed.shape[-1], np.arange(self.camera_embed.shape[-2]))
        self.camera_embed.data.copy_(torch.from_numpy(camera_embed).float().unsqueeze(0))
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def random_masking(self, x, mask_ratio):
        """
        Perform random masking by per-sample shuffling with independent masks per camera view.
        x: [B, N_c, P, D], sequence of patches with camera views separated
        mask_ratio: percentage of patches to mask
        """
        B, N_c, P, D = x.shape  # batch, cameras, patches, dim
        L = int(P * mask_ratio)  # number of patches to mask
        
        # 创建存储结果的容器
        x_masked = x.clone()
        mask = torch.zeros([B, N_c, P], device=x.device)
        ids_restore = torch.zeros([B, N_c, P], dtype=torch.long, device=x.device)
        
        # 为每个相机视角独立生成掩码
        for n in range(N_c):
            # 每个视角的patches
            x_view = x[:, n]  # [B, P, D]
            
            # 为每个视角独立生成随机噪声
            noise = torch.rand(B, P, device=x.device)  # uniform noise in [0, 1]
            
            # 排序得到掩码索引
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore_view = torch.argsort(ids_shuffle, dim=1)  # restore the original order
            
            # 生成二进制掩码: 0 is keep, 1 is remove
            mask_view = torch.ones([B, P], device=x.device)
            mask_view[:, :P-L] = 0
            
            # 重排得到最终掩码
            mask_view = torch.gather(mask_view, dim=1, index=ids_restore_view)
            
            # 为当前视角应用掩码
            mask_tokens = self.mask_token.expand(B, P, -1)
            x_masked[:, n] = x_view * (1.0 - mask_view.unsqueeze(-1)) + mask_tokens * mask_view.unsqueeze(-1)
            
            # 保存掩码和恢复索引
            mask[:, n] = mask_view
            ids_restore[:, n] = ids_restore_view
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        """
        Extract patch features from images and apply masking with independent masks per camera view
        x: [B, N_c, C, H, W] - batch of multi-view images
        Returns:
            x_masked: [B, N_c, P, D] - masked patch tokens
            mask: [B, N_c, P] - binary mask (1 is masked)
            ids_restore: [B, N_c, P] - indices for restoring original order
        """
        B, N_c = x.shape[:2]
        
        # Reshape for patch embedding
        x = rearrange(x, 'b n c h w -> (b n) c h w')
        
        # Extract patches - 只提取内容信息
        x = self.patch_embed(x)  # [B*N_c, P, D]

        x = rearrange(x, '(b n) p d -> b n p d', b=B, n=N_c)
        
        # 先应用随机掩码 - 只替换内容信息
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # 然后添加位置嵌入 - 保证所有token都有位置信息
        x_masked = x_masked + self.pos_embed # [B, N_c, P, D]
        
        # # 重排 x_masked 为 [B, N_c, P, D]
        # x_masked_reshaped = rearrange(x_masked, '(b n) p d -> b n p d', b=B, n=N_c)

        # 扩展相机嵌入
        camera_pos = self.camera_embed.unsqueeze(2)  # [1, N_c, 1, D]
        # 添加相机嵌入
        x_masked = x_masked + camera_pos

        
        return x_masked, mask, ids_restore
    
    def forward_decoder(self, x_masked, vision_tokens, future_step):
        """
        Decode masked patches using vision encoder tokens
        x_masked: [B, N_c, P, D] - masked patch tokens
        vision_tokens: [B, N_c+T, D_enc] - vision encoder tokens
        future_step: [B] - future time step for each sample
        Returns:
            pred: [B*N_c, P, patch_size*patch_size*3] - reconstructed patches
        """
        B, N_c, P, D = x_masked.shape
        
        # Project vision tokens from encoder dimension to decoder dimension
        vision_tokens = self.vision_projector(vision_tokens)  # [B, N_c+T, D]
        
        # Embed future timestep
        t_emb = self.future_timestep_embedder(future_step)  # [B, D]
        
        # # Expand timestep embedding for each camera view
        # t_emb = repeat(t_emb, 'b d -> (b n) d', n=self.num_cameras)  # [B*N_c, D]

        x_masked = rearrange(x_masked, 'b n p d -> b (n p) d')
        
        # Pass through transformer blocks
        for block in self.blocks:
            x_masked = block(x_masked, t_emb, vision_tokens) # [B, N_c*P, D]

        # x_masked = rearrange(x_masked, 'b (n p) d -> b n p d', n=self.num_cameras)
        
        # Reconstruct patches
        pred = self.final_layer(x_masked, t_emb)  # [B, N_c*P, patch_size*patch_size*3]

        pred = rearrange(pred, 'b (n p) pr -> b n p pr', n=self.num_cameras, p=self.num_patches) # pr: patch_resolution
        
        return pred
    
    def forward(self, x, vision_tokens, future_step):
        """
        Forward pass for masked image prediction
        x: [B, N_c, C, H, W] - future multi-view images to predict
        vision_tokens: [B, N_c+T, D_enc] - vision encoder tokens
        future_step: [B] - future time step for each sample in [1, max_future_step]
        Returns:
            loss: masked patch reconstruction loss
            pred: reconstructed patches
            mask: binary mask (1 is masked)
        """
        # First, encode images and apply masking
        x_masked, mask, ids_restore = self.forward_encoder(x)
        
        # Then decode masked patches
        pred = self.forward_decoder(x_masked, vision_tokens, future_step)
        
        # Calculate loss
        loss = self.compute_loss(x, pred, mask, ids_restore)
        
        return loss, pred, mask
    
    def compute_loss(self, x, pred, mask, ids_restore):
        """
        Compute the reconstruction loss for masked patches only
        """
        B, N_c, C, H, W = x.shape
        p = self.patch_size
        
        # 将掩码调整为适合损失计算的形状
        mask_flat = rearrange(mask, 'b n p -> (b n) p')

        # Reshape images into patches
        target = rearrange(x, 'b n c (h p1) (w p2) -> (b n) (h w) (p1 p2 c)', p1=p, p2=p)
        
        # Reshape predictions to match target
        pred = rearrange(pred, 'b n hw (p1 p2 c) -> (b n) hw (p1 p2 c)', p1=p, p2=p)
        
        # Compute loss only for masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B*N_c, P]
        
        # Apply mask: only compute loss on masked patches
        loss = (loss * mask_flat).sum() / mask_flat.sum()
        
        return loss
    
    def generate_images(self, vision_tokens, future_step, device):
        """
        Generate future images without masking (for inference)
        vision_tokens: [B, N_c+T, D_enc] - vision encoder tokens
        future_step: [B] - future time step for each sample
        Returns:
            pred_imgs: [B, N_c, 3, H, W] - generated images
        """
        B = vision_tokens.shape[0]
        N_c = self.num_cameras
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        
        # Replace all tokens with mask tokens
        mask_tokens = self.mask_token.expand(B * N_c, self.num_patches, -1)
        x = mask_tokens

        # Create empty patch tokens with positions
        x = x + self.pos_embed

        # Add camera embeddings
        camera_pos = repeat(self.camera_embed, '1 n d -> b n d', b=B)
        camera_pos = rearrange(camera_pos, 'b n d -> (b n) 1 d')
        x = x + camera_pos
        
        # Project vision tokens from encoder dimension to decoder dimension
        vision_tokens = self.vision_projector(vision_tokens)
        
        # Embed future timestep
        t_emb = self.future_timestep_embedder(future_step)  # [B, D]
        
        x = rearrange(x, '(b n) p d -> b (n p) d', b=B, n=N_c)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, vision_tokens)
        
        # Reconstruct patches
        pred = self.final_layer(x, t_emb) # [B, N_c*P, patch_size*patch_size*3]
        
        # Reshape to images
        pred_imgs = rearrange(pred, 'b (n h w) (p1 p2 c) -> b n c (h p1) (w p2)', 
                             n=N_c, h=h, w=w, p1=p, p2=p)
        
        return pred_imgs


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Generate 2D sine-cosine positional embedding.
    embed_dim: embedding dimension
    grid_size: int, grid height and width
    return: (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim) embeddings
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 2D sine-cosine positional embedding from grid.
    embed_dim: embedding dimension
    grid: 2D grid coordinates
    return: (grid_size*grid_size, embed_dim) embeddings
    """
    assert embed_dim % 2 == 0

    # Use half of dimensions to encode grid_h and half for grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sine-cosine positional embedding from grid.
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

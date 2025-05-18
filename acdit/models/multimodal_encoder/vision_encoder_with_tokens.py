import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, RmsNorm, PatchEmbed


def modulate(x, shift, scale):
    """
    Apply AdaLN modulation.
    x: (B, N, D) input features
    shift: (B, D) shift parameters
    scale: (B, D) scale parameters
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Attention Layers                                                #
#################################################################################

class Attention(nn.Module):
    """
    Masked multi-head attention module aligned with blocks.py implementation.
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.1, # TODO: make it in config
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
        """
        Forward pass for self-attention.
        x: (B, N, C) input tensor
        mask: optional attention mask
        Returns: (B, N, C) output tensor
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, H, N, D_h)
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
            attn = q @ k.transpose(-2, -1)  # (B, H, N, N)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # (B, H, N, D_h)

        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
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
            qkv_bias: bool = True,
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
#               Language Embedding Layer                                        #
#################################################################################

class LanguageEmbedder(nn.Module):
    """
    Projects language conditions into token space.
    """
    def __init__(self, hidden_size, embedding_dim=512):
        super().__init__()
        self.embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, hidden_size, bias=True)
        )

    def forward(self, language_embed):
        """
        Forward pass for language embedding.
        language_embed: (B, D_lang) language embedding
        Returns: (B, D) projected language embeddings
        """
        return self.embedding_projection(language_embed)

#################################################################################
#                                 Core Transformer Module                        #
#################################################################################

class TransformerBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning.
    """
    def __init__(self, hidden_size, 
                 num_heads, 
                 mlp_ratio=4.0, 
                 attn_drop=0.1,
                 proj_drop=0.1,
                 qk_norm=False,
                 mlp_drop=0.05,
                 **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, 
                              qkv_bias=True, attn_drop=attn_drop, 
                              proj_drop=proj_drop, qk_norm=qk_norm, 
                              **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, 
                       act_layer=approx_gelu, drop=mlp_drop) # TODO: make it in config
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask=None):
        """
        Forward pass with AdaLN modulation.
        x: (B, N, D) input features
        c: (B, D) conditioning embedding
        mask: optional attention mask
        Returns: (B, N, D) output features
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


#################################################################################
#                                 SpatioTemporalTokenFusion                     #
#################################################################################

# class SpatioTemporalTokenFusion(nn.Module):
#     def __init__(self, hidden_size, num_heads=8, attn_drop=0.1, proj_drop=0.1, qk_norm=False, fusion_type='cross_fusion'):
#         super().__init__()


#         self.fusion_type = fusion_type
#         if fusion_type == 'cross_fusion':
#             # 空间tokens作为查询，时间tokens作为键/值
#             self.space_attend_time = CrossAttention(
#                 dim=hidden_size, 
#                 num_heads=num_heads, 
#                 qkv_bias=True,
#                 qk_norm=qk_norm,
#                 attn_drop=attn_drop,
#                 proj_drop=proj_drop
#             )            
#             # 时间tokens作为查询，空间tokens作为键/值
#             self.time_attend_space = CrossAttention(
#                 dim=hidden_size, 
#                 num_heads=num_heads,
#                 qkv_bias=True,
#                 qk_norm=qk_norm,
#                 attn_drop=attn_drop,
#                 proj_drop=proj_drop
#             )

#             # # 独立的输出投影层
#             # self.space_projection = nn.Sequential(
#             #     nn.LayerNorm(hidden_size),
#             #     nn.Linear(hidden_size, hidden_size)
#             # )
            
#             # self.time_projection = nn.Sequential(
#             #     nn.LayerNorm(hidden_size),
#             #     nn.Linear(hidden_size, hidden_size)
#             # )

#             self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

#             self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 2 * hidden_size, bias=True)
#             )

#             self.fc = nn.Linear(hidden_size, hidden_size)


#         elif fusion_type == 'mixed_fusion':
#             self.norm = nn.LayerNorm(hidden_size)

#             self.mixed_attention = Attention(
#                 dim=hidden_size, 
#                 num_heads=num_heads,
#                 qkv_bias=True,
#                 qk_norm=qk_norm,
#                 attn_drop=attn_drop,
#                 proj_drop=proj_drop
#             )

#             self.shared_projection = nn.Sequential(
#                 nn.LayerNorm(hidden_size),
#                 nn.Linear(hidden_size, hidden_size)
#             )
#         else:
#             raise ValueError(f"Invalid fusion type: {fusion_type}")
        
#     def forward(self, spatial_tokens, temporal_tokens, c, mask=None):
#         """
#         Forward pass for SpatioTemporalTokenFusion.
#         spatial_tokens: [B, T, D]
#         temporal_tokens: [B, N_c, D]
#         c: (B, D) conditioning embedding
#         mask: optional attention mask
#         Returns: (B, N_c+T, D) sequence of visual goal representations from learnable tokens
#         """
#         if self.fusion_type == 'cross_fusion':
#             # 空间tokens关注时间tokens
#             enhanced_spatial = self.space_attend_time(
#                 spatial_tokens, temporal_tokens, mask)
                
#             # 时间tokens关注空间tokens
#             enhanced_temporal = self.time_attend_space(
#                 temporal_tokens, spatial_tokens, mask)
                
#             # 残差连接
#             spatial_tokens = spatial_tokens + enhanced_spatial
#             temporal_tokens = temporal_tokens + enhanced_temporal
            
#             # # 最终投影
#             # spatial_output = self.space_projection(spatial_tokens)
#             # temporal_output = self.time_projection(temporal_tokens)
            
#             # # 拼接结果
#             # output = torch.cat([temporal_output, spatial_output], dim=1)

#             all_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=1)
#             # 使用adaLN_modulation进行调节
#             shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
#             output = modulate(self.norm(all_tokens), shift, scale)
#             # 使用fc进行投影
#             output = self.fc(output)

#         elif self.fusion_type == 'mixed_fusion':
#             mixed_tokens = torch.cat([spatial_tokens, temporal_tokens], dim=1)
#             enhanced_mixed_tokens = self.mixed_attention(self.norm(mixed_tokens), mask)
#             mixed_tokens = mixed_tokens + enhanced_mixed_tokens
#             output = self.shared_projection(mixed_tokens)
#         else:
#             raise ValueError(f"Invalid fusion type: {self.fusion_type}")
#         return output

class SpatioTemporalTokenFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=8, attn_drop=0.1, proj_drop=0.1, qk_norm=False, fusion_type='cross_fusion'):
        super().__init__()
        
        # 空间tokens作为查询，时间tokens作为键/值
        self.space_attend_time = CrossAttention(
            dim=hidden_size, 
            num_heads=num_heads, 
            qkv_bias=True,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )            
        # 时间tokens作为查询，空间tokens作为键/值
        self.time_attend_space = CrossAttention(
            dim=hidden_size, 
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )

        # 独立的输出投影层
        self.space_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.time_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, spatial_tokens, temporal_tokens, mask=None):
        # spatial_tokens: [B, T, D]
        # temporal_tokens: [B, N_c, D]
        
        # 空间tokens关注时间tokens
        enhanced_spatial = self.space_attend_time(
            spatial_tokens, temporal_tokens, mask)
            
        # 时间tokens关注空间tokens
        enhanced_temporal = self.time_attend_space(
            temporal_tokens, spatial_tokens, mask)
            
        # 残差连接
        spatial_tokens = spatial_tokens + enhanced_spatial
        temporal_tokens = temporal_tokens + enhanced_temporal
        
        # 最终投影
        spatial_output = self.space_projection(spatial_tokens)
        temporal_output = self.time_projection(temporal_tokens)
        
        # 拼接结果
        output = torch.cat([temporal_output, spatial_output], dim=1)
        
        return output


#################################################################################
#                    Vision Encoder Model with Learnable Tokens                 #
#################################################################################

class MVT_TokenFusion_Encoder(nn.Module):
    """
    MVT (Multi-View Temporal) Token Fusion Encoder with learnable tokens for both spatial and temporal attention.
    
    This encoder processes multi-view video inputs using a novel attention framework:
    1. Input: (B, N_c, T, C, H, W) -> Patch embedding -> (B, N_c, T, P, D)
    2. Learnable tokens: N_c + T tokens of dimension D
    3. Cross-view spatial MHSA: Reshape to (B*T, P*N_c, D), concatenate with T spatial tokens
    4. Cross-region temporal MHSA: Reshape to (B*N_c, P*T, D), concatenate with N_c temporal tokens
    5. Alternate between spatial and temporal attention
    6. Output: Collection of N_c+T tokens as context information
    """
    def __init__(
        self,
        input_size=224,              # Input image size
        patch_size=16,               # Image patch size
        in_channels=3,               # Input channels
        hidden_size=512,             # Hidden feature dimension
        depth=10,                    # Number of transformer layers
        num_heads=8,                # Number of attention heads
        mlp_ratio=4.0,               # MLP ratio
        num_frames=2,                # Number of time frames
        num_cameras=2,               # Number of cameras
        language_dim=512,            # Language embedding dimension
        output_dim=None,             # Output dimension, defaults to hidden_size
        attn_drop=0.1,               # Attention dropout
        proj_drop=0.1,               # Projection dropout
        qk_norm=False,               # Query-key normalization
        use_independent_patch_embed=False, # Use independent patch embedding for each camera # new added
        use_token_fusion=False,       # Use token fusion module to replace the original output projection # new added
        fusion_type='cross_fusion',   # Fusion type, 'cross_fusion' or 'mixed_fusion'
        cross_attn_drop=0.1,         # Cross-attention dropout
        cross_proj_drop=0.1,         # Cross-projection dropout
        cross_qk_norm=False,          # Cross-query-key normalization
        mlp_drop=0.05,               # MLP dropout
        shared_language_projection=False, # new added
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.output_dim = output_dim if output_dim is not None else hidden_size
        self.use_token_fusion = use_token_fusion
        self.fusion_type = fusion_type
        self.use_independent_patch_embed = use_independent_patch_embed
        self.shared_language_projection = shared_language_projection
        self.num_patches = None
        # Image patch embedding
        if use_independent_patch_embed:
            self.x_embedders = nn.ModuleList([
                PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
                for _ in range(num_cameras)
            ])
            self.num_patches = self.x_embedders[0].num_patches
        else:
            self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
            self.num_patches = self.x_embedder.num_patches
        
        # Language embedding projection
        self.language_projection = None
        if not shared_language_projection:
            self.language_embedder = LanguageEmbedder(hidden_size, language_dim)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.camera_embed = nn.Parameter(torch.zeros(1, num_cameras, hidden_size), requires_grad=False)
        
        # Learnable tokens: spatial tokens (T) and temporal tokens (N_c)
        self.spatial_tokens = nn.Parameter(torch.zeros(1, num_frames, hidden_size))
        self.temporal_tokens = nn.Parameter(torch.zeros(1, num_cameras, hidden_size))
        
        self.hidden_size = hidden_size

        # Transformer blocks alternating spatial and temporal attention
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                             attn_drop=attn_drop, proj_drop=proj_drop, 
                             qk_norm=qk_norm, mlp_drop=mlp_drop) 
            for _ in range(depth)
        ])

        if use_token_fusion:
            # 添加token交互模块, 替代原来的输出映射
            self.token_fusion = SpatioTemporalTokenFusion(hidden_size, num_heads=num_heads, 
                                                      attn_drop=cross_attn_drop, 
                                                      proj_drop=cross_proj_drop, 
                                                      qk_norm=cross_qk_norm,
                                                      fusion_type=fusion_type)
        else:
            # Output projection for each token
            self.output_projection = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, self.output_dim)
            )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed with sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize temporal position embedding
        temp_embed = get_1d_sincos_pos_embed_from_grid(self.temp_embed.shape[-1], np.arange(self.temp_embed.shape[-2]))
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))
        
        # Initialize camera position embedding
        camera_embed = get_1d_sincos_pos_embed_from_grid(self.camera_embed.shape[-1], np.arange(self.camera_embed.shape[-2]))
        self.camera_embed.data.copy_(torch.from_numpy(camera_embed).float().unsqueeze(0))

        # Initialize learnable tokens
        nn.init.normal_(self.spatial_tokens, std=0.02)
        nn.init.normal_(self.temporal_tokens, std=0.02)
        
        # Initialize patch_embed like nn.Linear
        if self.use_independent_patch_embed:
            for i in range(self.num_cameras):
                w = self.x_embedders[i].proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(self.x_embedders[i].proj.bias, 0)
        else:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize language embedding MLP
        if not self.shared_language_projection and self.language_embedder is not None:
            nn.init.normal_(self.language_embedder.embedding_projection[-1].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                      
        if self.use_token_fusion and hasattr(self, 'token_fusion'):
            if self.fusion_type == 'cross_fusion':
                nn.init.normal_(self.token_fusion.space_projection[1].weight, std=0.02)
                if self.token_fusion.space_projection[1].bias is not None:
                    nn.init.constant_(self.token_fusion.space_projection[1].bias, 0)
                nn.init.normal_(self.token_fusion.time_projection[1].weight, std=0.02)
                if self.token_fusion.time_projection[1].bias is not None:
                    nn.init.constant_(self.token_fusion.time_projection[1].bias, 0)
                # # 初始化token_fusion中的各个投影层
                # nn.init.normal_(self.token_fusion.adaLN_modulation[1].weight, std=0.02)
                # nn.init.constant_(self.token_fusion.adaLN_modulation[1].bias, 0)
                # nn.init.normal_(self.token_fusion.fc.weight, std=0.02)
                # nn.init.constant_(self.token_fusion.fc.bias, 0)
            elif self.fusion_type == 'mixed_fusion':
                # nn.init.normal_(self.token_fusion.mixed_attention.proj.weight, std=0.02)
                # if self.token_fusion.mixed_attention.proj.bias is not None:
                #     nn.init.constant_(self.token_fusion.mixed_attention.proj.bias, 0)
                nn.init.normal_(self.token_fusion.shared_projection[1].weight, std=0.02)
                if self.token_fusion.shared_projection[1].bias is not None:
                    nn.init.constant_(self.token_fusion.shared_projection[1].bias, 0)
        else:
            # Initialize output projection
            nn.init.normal_(self.output_projection[1].weight, std=0.02)
            nn.init.constant_(self.output_projection[1].bias, 0)

    def forward(self, x, y):
        """
        Forward pass of CrossViewTemporalTokenEncoder.
        x: (B, N_c, T, C, H, W) multi-view video input, where N_c is number of cameras, T is number of time steps
        y: (B, D_lang) language condition embedding
        Returns: (B, N_c+T, D) sequence of visual goal representations from learnable tokens
        """
        B, N_c, T, C, H, W = x.shape
        assert N_c == self.num_cameras, f"Input camera count {N_c} doesn't match configured {self.num_cameras}"
        assert T == self.num_frames, f"Input frame count {T} doesn't match configured {self.num_frames}"
        P = self.num_patches
        
        # Language condition embedding
        if not self.shared_language_projection and self.language_embedder is not None:
            y_embed = self.language_embedder(y)  # (B, D)
        else:
            y_embed = y # (B, D)

        # Prepare language conditions for spatial and temporal attention blocks
        y_spatial = repeat(y_embed, 'b d -> (b t) d', t=T)  # (B*T, D)
        y_temporal = repeat(y_embed, 'b d -> (b n) d', n=N_c)  # (B*N_c, D)
        
        # Reshape for patch embedding & Reshape back
        if self.use_independent_patch_embed:
            x = rearrange(x, 'b n t c h w -> (b t) n c h w')  # (B*T, N_c, C, H, W)
            x = torch.cat([self.x_embedders[i](x[:, i]).unsqueeze(1) for i in range(N_c)], dim=1)  # (B*T, N_c, P, D), P is patch count
            x = rearrange(x, '(b t) n p d -> b n t p d', b=B, n=N_c, t=T)  # (B, N_c, T, P, D)
        else:
            x = rearrange(x, 'b n t c h w -> (b n t) c h w')  # (B*N_c*T, C, H, W)
            x = self.x_embedder(x)                            # (B*N_c*T, P, D), P is patch count
            x = rearrange(x, '(b n t) p d -> b n t p d', b=B, n=N_c, t=T)  # (B, N_c, T, P, D)
        
        # Add spatial position embedding to each patch
        x = x + self.pos_embed.unsqueeze(1).unsqueeze(1)  # Add spatial pos embed
        
        # Add camera position encoding to each camera view
        cam_pos = rearrange(self.camera_embed, '1 n d -> 1 n 1 1 d')  # (1, N_c, 1, 1, D)
        x = x + cam_pos  # (B, N_c, T, P, D)
        
        # Prepare spatial and temporal tokens (expand batch dimension)
        spatial_tokens = repeat(self.spatial_tokens, '1 t d -> b t d', b=B)  # (B, T, D)
        temporal_tokens = repeat(self.temporal_tokens, '1 n d -> b n d', b=B)  # (B, N_c, D)
        
        # Process with alternating spatial and temporal transformer blocks
        for i in range(0, len(self.blocks), 2):
            if i + 1 >= len(self.blocks):
                break
                
            spatial_block, temp_block = self.blocks[i], self.blocks[i+1]
            
            # --- Cross-view spatial attention (process different camera views at the same timestep) ---
            
            # 1. Reshape data for each timestep
            x_spatial = rearrange(x, 'b n t p d -> (b t) (n p) d')  # (B*T, N_c*P, D)
            
            # 2. Prepare spatial token for each timestep
            st = rearrange(spatial_tokens, 'b t d -> (b t) d').unsqueeze(1)  # (B*T, 1, D)
            
            # 3. Concatenate data with corresponding token
            x_with_st = torch.cat([x_spatial, st], dim=1)  # (B*T, N_c*P+1, D)
            
            # 4. Spatial attention processing
            x_with_st = spatial_block(x_with_st, y_spatial)  # (B*T, N_c*P+1, D)
            
            # 5. Separate updated token and data
            x_spatial = x_with_st[:, :-1, :]  # (B*T, N_c*P, D)
            st = x_with_st[:, -1:, :]  # (B*T, 1, D)
            
            # 6. Update spatial tokens
            spatial_tokens = rearrange(st.squeeze(1), '(b t) d -> b t d', b=B)  # (B, T, D)
            
            # 7. Reshape data back to original shape
            x = rearrange(x_spatial, '(b t) (n p) d -> b n t p d', b=B, n=N_c, p=P)  # (B, N_c, T, P, D)
            
            # --- Cross-region temporal attention (process different timesteps for each camera view) ---
             
            # 1. Add temporal position encoding only before the first temporal attention block
            if i == 0:
                temp_pos = rearrange(self.temp_embed, '1 t d -> 1 1 t 1 d')  # (1, 1, T, 1, D)
                x = x + temp_pos  # Add temporal position encoding

            
            # 2. Reshape data for each camera view
            x_temporal = rearrange(x, 'b n t p d -> (b n) (t p) d')  # (B*N_c, T*P, D)
            
            # 3. Prepare temporal token for each camera view
            tt = rearrange(temporal_tokens, 'b n d -> (b n) d').unsqueeze(1)  # (B*N_c, 1, D)
            
            # 4. Concatenate data with corresponding token
            x_with_tt = torch.cat([x_temporal, tt], dim=1)  # (B*N_c, T*P+1, D)
            
            # 5. Temporal attention processing
            x_with_tt = temp_block(x_with_tt, y_temporal)  # (B*N_c, T*P+1, D)
            
            # 6. Separate updated token and data
            x_temporal = x_with_tt[:, :-1, :]  # (B*N_c, T*P, D)
            tt = x_with_tt[:, -1:, :]  # (B*N_c, 1, D)
            
            # 7. Update temporal tokens
            temporal_tokens = rearrange(tt.squeeze(1), '(b n) d -> b n d', b=B)  # (B, N_c, D)
            
            # 8. Reshape data back to original shape
            x = rearrange(x_temporal, '(b n) (t p) d -> b n t p d', b=B, n=N_c, t=T)  # (B, N_c, T, P, D)
        
        # Extract all learned tokens
        # Spatial tokens represent cross-view information at each timestep
        # Temporal tokens represent cross-time information for each camera view

        # spatial_tokens_output = self.output_projection(spatial_tokens)  # (B, T, D)
        # temporal_tokens_output = self.output_projection(temporal_tokens)  # (B, N_c, D)
        
        # # Combine all tokens as final output
        # output = torch.cat([
        #     rearrange(temporal_tokens_output, 'b n d -> b n d'),  # (B, N_c, D)
        #     rearrange(spatial_tokens_output, 'b t d -> b t d')    # (B, T, D)
        # ], dim=1)  # (B, N_c+T, D)

        # updated: use token fusion module
        if self.use_token_fusion:
            output = self.token_fusion(spatial_tokens, temporal_tokens)
        else:
            spatial_tokens_output = self.output_projection(spatial_tokens)
            temporal_tokens_output = self.output_projection(temporal_tokens)
            output = torch.cat([temporal_tokens_output, spatial_tokens_output], dim=1)
        
        return output


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





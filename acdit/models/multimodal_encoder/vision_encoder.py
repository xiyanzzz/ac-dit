import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed


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
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
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
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
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
#                              Vision Encoder Model                             #
#################################################################################

class VisionEncoder(nn.Module):
    """
    Multi-view vision encoder based on Latte architecture,
    designed specifically for extracting visual representations for robotic tasks.
    
    The encoder outputs a sequence of embeddings, with sequence length equal to
    the number of cameras, suitable for cross-attention in action policy models.
    """
    def __init__(
        self,
        input_size=224,              # Input image size
        patch_size=16,               # Image patch size
        in_channels=3,               # Input channels
        hidden_size=768,             # Hidden feature dimension
        depth=12,                    # Number of transformer layers
        num_heads=12,                # Number of attention heads
        mlp_ratio=4.0,               # MLP ratio
        num_frames=2,                # Number of time frames
        num_cameras=2,               # Number of cameras
        language_dim=512,            # Language embedding dimension
        output_dim=None,             # Output dimension, defaults to hidden_size
        qk_norm=False,               # Whether to normalize query and key
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.output_dim = output_dim if output_dim is not None else hidden_size

        # Image patch embedding
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        # Language embedding projection
        self.language_embedder = LanguageEmbedder(hidden_size, language_dim)

        # Calculate number of patches
        num_patches = self.x_embedder.num_patches
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.camera_embed = nn.Parameter(torch.zeros(1, num_cameras, hidden_size), requires_grad=False)
        
        self.hidden_size = hidden_size

        # Transformer blocks alternating spatial and temporal attention
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm) for _ in range(depth)
        ])

        # Output projection for each camera view
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, self.output_dim)
            ) for _ in range(num_cameras)
        ])

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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize temporal position embedding
        temp_embed = get_1d_sincos_pos_embed_from_grid(self.temp_embed.shape[-1], np.arange(self.temp_embed.shape[-2]))
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))
        
        # Initialize camera position embedding
        camera_embed = get_1d_sincos_pos_embed_from_grid(self.camera_embed.shape[-1], np.arange(self.camera_embed.shape[-2]))
        self.camera_embed.data.copy_(torch.from_numpy(camera_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize language embedding MLP
        nn.init.normal_(self.language_embedder.embedding_projection[-1].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        # Initialize output projections
        for proj in self.output_projections:
            nn.init.normal_(proj[1].weight, std=0.02)
            nn.init.constant_(proj[1].bias, 0)

    def forward(self, x, y):
        """
        Forward pass of VisionEncoder.
        x: (B, N_c, T, C, H, W) multi-view video input, where N_c is number of cameras, T is number of time steps
        y: (B, D_lang) language condition embedding
        Returns: (B, N_c, D) sequence of visual goal representations, sequence length = num_cameras
        """
        B, N_c, T, C, H, W = x.shape
        assert N_c == self.num_cameras, f"Input camera count {N_c} doesn't match configured {self.num_cameras}"
        assert T == self.num_frames, f"Input frame count {T} doesn't match configured {self.num_frames}"
        
        # Language condition embedding
        y_embed = self.language_embedder(y)  # (B, D)
        
        # Reshape input for patch embedding
        x = rearrange(x, 'b n t c h w -> (b n t) c h w')  # (B*N_c*T, C, H, W)
        x = self.x_embedder(x)                            # (B*N_c*T, P, D), P is patch count
        x = x + self.pos_embed                            # Add spatial position encoding (once)
        
        # Add camera position encoding (once)
        x = rearrange(x, '(b n t) p d -> (b t) n p d', b=B, n=N_c, t=T)  # (B*T, N_c, P, D)
        x = x + self.camera_embed.unsqueeze(2)  # Broadcast camera position encoding: (1, N_c, 1, D) -> (B*T, N_c, P, D)
        x = rearrange(x, '(b t) n p d -> (b n t) p d', b=B, n=N_c)  # Back to shape (B*N_c*T, P, D)
        
        # Prepare language condition for spatial and temporal attention blocks
        # These remain constant throughout the transformer layers
        y_spatial = repeat(y_embed, 'b d -> (b t) d', t=T)                         # (B*T, D)
        y_temp = repeat(y_embed, 'b d -> (b n p) d', n=N_c, p=self.x_embedder.num_patches)  # (B*N_c*P, D)
        
        # Process with alternating spatial and temporal transformer blocks
        for i in range(0, len(self.blocks), 2):
            if i + 1 >= len(self.blocks):
                break
                
            spatial_block, temp_block = self.blocks[i], self.blocks[i+1]
            
            # Reshape to fuse camera views and patches for spatial attention
            # This allows attention between patches across different cameras at the same time frame
            x = rearrange(x, '(b n t) p d -> (b t) (n p) d', b=B, n=N_c, t=T)  # (B*T, N_c*P, D)
            
            # Spatial attention block (operating across all cameras for each time step)
            x = spatial_block(x, y_spatial)  # (B*T, N_c*P, D)
            
            # Reshape for temporal attention
            x = rearrange(x, '(b t) (n p) d -> (b n p) t d', b=B, n=N_c, t=T, p=self.x_embedder.num_patches)  # (B*N_c*P, T, D)
            
            # Add temporal position encoding only before the first temporal attention block
            if i == 0:
                x = x + self.temp_embed  # Add temporal position encoding (once)
            
            # Temporal attention block
            x = temp_block(x, y_temp)  # (B*N_c*P, T, D)
            
            # Reshape back to original format
            x = rearrange(x, '(b n p) t d -> (b n t) p d', b=B, n=N_c, t=T, p=self.x_embedder.num_patches)  # (B*N_c*T, P, D)
        
        # Feature aggregation - reshape directly as suggested
        x = rearrange(x, '(b n t) p d -> b n (t p) d', b=B, n=N_c, t=T)  # (B, N_c, T*P, D)

        camera_features = []
        for cam_idx in range(N_c):
            # Extract and average features for this camera
            cam_features = x[:, cam_idx]  # (B, T*P, D)
            cam_features = torch.mean(cam_features, dim=1)  # (B, D)
            
            # Apply output projection
            cam_output = self.output_projections[cam_idx](cam_features)  # (B, output_dim)
            camera_features.append(cam_output)

        # Stack camera features to create sequence output
        output = torch.stack(camera_features, dim=1)  # (B, N_c, output_dim)
        
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

#################################################################################
#                                 Model Configurations                          #
#################################################################################

# def VisionEncoder_L(**kwargs):
#     """Large vision encoder model configuration"""
#     return VisionEncoder(depth=24, hidden_size=1024, num_heads=16, **kwargs)

# def VisionEncoder_B(**kwargs):
#     """Base vision encoder model configuration"""
#     return VisionEncoder(depth=12, hidden_size=768, num_heads=12, **kwargs)

# def VisionEncoder_S(**kwargs):
#     """Small vision encoder model configuration"""
#     return VisionEncoder(depth=12, hidden_size=384, num_heads=6, **kwargs)

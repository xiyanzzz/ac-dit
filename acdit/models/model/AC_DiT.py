import torch
import torch.nn as nn
import numpy as np

from AC_DiT.models.model.blocks import (FinalLayer, AC_DiTBlock, TimestepEmbedder,
                               get_1d_sincos_pos_embed_from_grid,
                               LanguageEmbedder)

class AC_DiT(nn.Module):
    def __init__(
            self,
            action_dim = 7,
            hidden_dim = 512,
            obs_horizon = 2,
            pred_horizon = 10,
            num_heads = 8,
            mlp_ratio = 4,
            num_layers = 8,
            attn_drop = 0.1,                # Attention dropout
            proj_drop = 0.1,                # Projection dropout
            qk_norm = False,                # Query-key normalization
            cross_attn_drop = 0.3,          # Cross-attention dropout
            cross_proj_drop = 0.1,          # Cross-projection dropout
            cross_qk_norm = True,           # Cross-query-key normalization
            mlp_drop = 0.05,                # MLP dropout
            shared_language_projection = False, # new added
            mlp_embedder = False, # new added
            linear_output = False, # new added
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.shared_language_projection = shared_language_projection
        self.mlp_embedder = mlp_embedder
        self.linear_output = linear_output
        if mlp_embedder:
            self.x_embedder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.x_embedder = nn.Linear(action_dim, hidden_dim)
        self.x_pos_embed = nn.Parameter(torch.zeros(1, pred_horizon+obs_horizon, hidden_dim))
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.y_embedder = None
        if not shared_language_projection:
            self.y_embedder = LanguageEmbedder(hidden_dim) # 512 is the output dim of the language model

        self.blocks = nn.ModuleList([
            AC_DiTBlock(hidden_dim, num_heads, mlp_ratio, attn_drop, proj_drop, qk_norm, cross_attn_drop, cross_proj_drop, cross_qk_norm, mlp_drop) for _ in range(num_layers)
        ])

        self.final_layer = FinalLayer(hidden_dim, action_dim, linear_output)

        self.initialize_weights()

        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        # Initialize x_embedder
        if self.mlp_embedder:
            nn.init.xavier_uniform_(self.x_embedder[0].weight)
            nn.init.constant_(self.x_embedder[0].bias, 0)
            nn.init.xavier_uniform_(self.x_embedder[2].weight)
            nn.init.constant_(self.x_embedder[2].bias, 0)
        else:
            nn.init.xavier_uniform_(self.x_embedder.weight)
            nn.init.constant_(self.x_embedder.bias, 0)
        _, pos, h = self.x_pos_embed.shape
        x_pos_embed = get_1d_sincos_pos_embed_from_grid(h,np.arange(pos, dtype=np.float32))
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize y_embedder
        if not self.shared_language_projection and self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_projection.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        if self.linear_output:
            nn.init.constant_(self.final_layer.linear.weight, 0) # for linear output
            nn.init.constant_(self.final_layer.linear.bias, 0)
        else:
            nn.init.constant_(self.final_layer.non_linear.fc2.weight, 0) # for non-linear output
            nn.init.constant_(self.final_layer.non_linear.fc2.bias, 0)

        
        
    def forward(self, x, x_cond, y, t, g):
        """
        Forward pass for the AC_DiT noise prediction network

        Args:
            x: (B, T_a, d_action) tensor of noise action
            x_cond: (B, T_o, d_action) tensor of conditioned action
            t: (B,) tensor of diffusion timesteps
            y: (B, d_clip) tensor of language condition tokens
            g: (B, T_g, D) tensor of goal-conditioned latent state
        """
        x = torch.cat([x_cond, x], dim=1)
        x = self.x_embedder(x) + self.x_pos_embed # (B, T, D) where T = T_a + T_o

        t = self.t_embedder(t) # (B,) -> (B, D) 
        if not self.shared_language_projection and self.y_embedder is not None:
            y = self.y_embedder(y) # (B, d_clip) -> (B, D)
        else:
            y = y # (B, D)
        c = t + y

        for block in self.blocks:
            x = block(x, c, g)

        x = self.final_layer(x, c)

        # Only preserve the predicted action tokens
        x = x[:, -self.pred_horizon:]

        return x


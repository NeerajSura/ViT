import torch
import torch.nn as nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.n_heads = config['n_heads']
        self.head_dim = config['head_dim']
        self.emb_dim = config['emb_dim']
        self.drop_prob = config['dropout'] if 'dropout' in config else 0.0
        self.attn_dim = self.n_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.emb_dim, 3 * self.attn_dim, bias=False)
        self.output_proj = nn.Sequential(
            nn.Linear(self.attn_dim, self.emb_dim),
            nn.Dropout(self.drop_prob)
        )

        self.attn_dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        #####################Part 1 : Project from D and split q k v##################################
        # Converting into attention dimension
        # Batch_size x num_patches x patch_dim
        B, N = x.shape[:2]

        # Project from D and split q k v
        # qkv proj -> Batch_size x n_patches x 3*attn_dim
        # each q(k and v) -> batch_size x n_patches x attn_dim
        q, k, v = self.qkv_proj(x).split(self.attn_dim, dim=-1)
        ##############################################################################################


        ####################Part2 : Split heads and compute attention representations#################
        # Batch Size x Number of Patches x Attention Dimension
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Heads x Number of Patches x Head Dimension
        # -> B x H x N x Head Dimension
        q = rearrange(q, 'b n (n_h h_d) -> b n_h n h_d', n_h=self.n_heads, h_d=self.head_dim)
        k = rearrange(k, 'b n (n_h h_d) -> b n_h n h_d', n_h=self.n_heads, h_d=self.head_dim)
        v = rearrange(v, 'b n (n_h h_d) -> b n_h n h_d', n_h=self.n_heads, h_d=self.head_dim)

        # Compute Q@K.t
        # B x H x N x Head Dimension @ B x H x Head Dimension x N
        # -> B x H x N x N
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**(-0.5))
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        ###############################################################################################

        ###################Part3 : Compute weighted value and project back to D########################
        # B x H x N x N @ B x H x N x HD
        # -> B x H x N x HD
        out = torch.matmul(attn, v)

        # Project back to D
        # B x H x N x HD --> B x N x H x Hd i.e B x N x attn_dim
        out = rearrange(out, 'b n_h n h_d -> b n (n_h h_d)')
        # B x N x D
        out = self.output_proj(out)

        return out
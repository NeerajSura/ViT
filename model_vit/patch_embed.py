import torch
import torch.nn as nn
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    r'''
    Layer to take the input image and do the following:
    1. Transform grid of image into sequence of patches
        Number of patches are decided based on the image height width and
        patch height width
    2. Add cls token to the above sequence of patches in the first position
    3. Add positional embedding to the above sequence(after adding cls)
    4. Dropout if needed
    '''

    def __init__(self, config):
        super(PatchEmbedding, self).__init__()
        # Example configuration
        # Image c, h, w : 3, 224, 224
        # Patch h, w : 16, 16
        im_height = config['image_height']
        im_width = config['image_width']
        im_channels = config['im_channels']
        emb_dim = config['emb_dim']
        patch_embd_drop = config['patch_emb_drop']

        self.patch_height = config['patch_height']
        self.patch_width = config['patch_width']

        # Compute number of patches for positional parameter initialization
        num_patches = (im_height // self.patch_height) * (im_width // self.patch_width)

        # This is the inout dimension of the patch embedding layer
        # After patchifying the 224, 224, 3 image will be num_patches * patch_h * patch_w * 3
        # which will be 196 * 16 * 16 * 3
        # Hence patch dimension will be 16 * 16 * 3
        patch_dim = im_channels * self.patch_width * self.patch_height

        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        # cls token needs to be added which needs positional embedding as well, hence num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(emb_dim))
        self.patch_emb_dropout = nn.Dropout(patch_embd_drop)

    def forward(self, x):
        batch_size = x.shape[0]

        # This is doing the B, 3, 224, 224 to (B, num_patches, patch_dim) transformation
        # B, 3, 224, 224 -> B, 3, 14*16, 14*16
        # B, 3, 14*16, 14*16 -> B, 3, 14, 16, 14, 16,
        # B, 3, 14, 16, 14, 16 -> B, 14, 14, 16, 16, 3
        # B, 14*14, 16*16*3 -> B, num_patches, patch_dim

        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                        ph=self.patch_height,
                        pw=self.patch_width)
        out = self.patch_embed(out)

        # Add cls
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=batch_size)
        out = torch.cat((cls_tokens, out), dim=1)

        # Add position embedding and do dropout
        out += self.pos_embed
        out = self.patch_emb_dropout(out)

        return out

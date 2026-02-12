import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttnPool2d(nn.Module):
    def __init__(self, C: int, attn_dim: int = 512, gated: bool = True, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=C)
        self.gated = gated
        self.theta = nn.Conv2d(C, attn_dim, kernel_size=1, bias=True)
        self.gate  = nn.Conv2d(C, attn_dim, kernel_size=1, bias=True) if gated else None
        self.score = nn.Conv2d(attn_dim, 1, kernel_size=1, bias=True)
        self.drop  = nn.Dropout(dropout)
    
    def forward(self, feats: torch.Tensor):
        x = self.norm(feats)                               # [B,C,H,W]
        h = torch.tanh(self.theta(x))                      # [B,A,H,W]
        if self.gated:
            h = h * torch.sigmoid(self.gate(x))           # gated tanh
        h = self.drop(h)
        logits = self.score(h)                             # [B,1,H,W]
        attn = torch.softmax(logits.flatten(2), dim=-1)    # [B,1,H*W]
        attn = attn.view(logits.shape)                     # [B,1,H,W]
        pooled = (feats * attn).sum(dim=(2,3))             # [B,C]
        return pooled, attn


class DINOModelWrapper(nn.Module):
    def __init__(self,
                 backbone,           
                 num_classes: int = 6,
                 img_size: int = 256,
                 proj_dim: int = 512,
                 attn_dim: int = 512,
                 head_hidden: int = 512,
                 head_dropout: float = 0.1,
                 pixel_mean_std=None,
                 embedding_dim=1024, #This is the dino embed dim, for vitl it is 1024
                 use_registers: bool = False): 
        super().__init__()

        self.encoder = backbone
        self.img_size = img_size
        self.use_registers = use_registers
        self.patch_size = self.encoder.patch_size
        self.C = embedding_dim

        assert img_size % self.patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({self.patch_size})"
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.grid_h = img_size // self.patch_size
        self.grid_w = img_size // self.patch_size
        
        self.pool = SpatialAttnPool2d(C=self.C, attn_dim=attn_dim, gated=True, dropout=head_dropout)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.C), 
            nn.Linear(self.C, proj_dim), 
            nn.GELU(),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes)
        )
        if pixel_mean_std is not None:
            mean, std = pixel_mean_std
            self.register_buffer('pixel_mean', torch.tensor(mean).view(1, -1, 1, 1))
            self.register_buffer('pixel_std', torch.tensor(std).view(1, -1, 1, 1))
        else:
            self.pixel_mean = None
            self.pixel_std = None

    def _resize_to_img_size(self, x):
        """Resize input to the expected image size."""
        if x.shape[-2] == self.img_size and x.shape[-1] == self.img_size:
            return x
        return F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

    def _apply_pixel_norm(self, x):
        """Apply pixel-wise normalization if configured."""
        if self.pixel_mean is None:
            return x
        return (x - self.pixel_mean) / self.pixel_std

    def _reshape_patch_tokens_to_spatial(self, patch_tokens):
        B, N, C = patch_tokens.shape
        H, W = self.grid_h, self.grid_w
        if N != H * W:
            raise RuntimeError(
                f"Number of patch tokens ({N}) doesn't match expected grid size "
                f"({H}x{W}={H*W}). Check if registers are being included."
            )
        # Reshape: [B, num_patches, C] -> [B, H, W, C] -> [B, C, H, W]
        spatial_feats = patch_tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return spatial_feats

    def extract_features(self, x):
        if hasattr(self.encoder, 'forward_features'):
            output = self.encoder.forward_features(x)
            
            if isinstance(output, dict):
                if 'x_norm_patchtokens' in output:
                    patch_tokens = output['x_norm_patchtokens']
                else:
                    raise KeyError(f"Unexpected DINOv2 output keys: {output.keys()}")
        else:
            print('Using Fallback for older implementations. Assuming [B, 1+N, C] with CLS token first')
            output = self.encoder(x)
            patch_tokens = output[:, 1:, :]
        
        return patch_tokens #[B, num_patches, C]

    def forward(self, x: torch.Tensor, return_attn: bool = False):

        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {x.shape[1]}")
        x = self._resize_to_img_size(x)
        x = self._apply_pixel_norm(x)
        
        patch_tokens = self.extract_features(x)  # [B, N, C]
        spatial_feats = self._reshape_patch_tokens_to_spatial(patch_tokens)  # [B, C, H, W]
        pooled, attn = self.pool(spatial_feats)  # [B, C], [B, 1, H, W]
        emb = self.proj(pooled)  # [B, proj_dim]
        logits = self.head(emb)  # [B, num_classes]
        
        if return_attn:
            return logits, emb, attn, spatial_feats
        else:
            return logits, emb
        
if __name__ == "__main__":
    DINOV3_LIBRARY_PATH = '/home/obed/projects/aip-medilab/obed/medproj/dinov3'
    DINOV3_CHECKPOINTS_PATH='/datasets/exactvu_pca/checkpoint_store'

    def dinov3_vitl16(**kwargs):
        import torch
        import os
        REPO_DIR = DINOV3_LIBRARY_PATH
        weight_basename = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        weights = os.path.join(DINOV3_CHECKPOINTS_PATH, weight_basename)
        if not os.path.exists(weights):
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {weights}")
        kwargs["weights"] = weights
        model = torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", **kwargs)
        return model
    
    model = dinov3_vitl16()

    dmodel = DINOModelWrapper(
        dinov3_model=model,
        num_classes=6,
        img_size=256,
        proj_dim=512,
        attn_dim=512,
        head_hidden=512,
        head_dropout=0.1,
        pixel_mean_std=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )
    x = torch.randn(4, 3, 256, 256)
    
    logits, embeddings = dmodel(x)
    print(f"Logits shape: {logits.shape}")  # [B, num_classes]
    print(f"Embeddings shape: {embeddings.shape}")  # [B, proj_dim]

    logits, embeddings, attn, feats = dmodel(x, return_attn=True)
    print(f"Attention shape: {attn.shape}")  # [B, 1, H, W]
    print(f"Features shape: {feats.shape}")  # [B, dino_embed_dim, H, W]
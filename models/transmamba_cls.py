# TransMamba-Cls: Hybrid Transformer + Mamba for Text Classification
# Based on: Zhu et al. "TransMamba" (2025) — adapted for GLUE text classification
#
# Improvements (aligned with paper):
# 1. Fusion module theo đúng paper: Feature Projection trước Cross-Attention ✅
# 2. Tăng Mamba decoder: 8 layers (paper: 16L) ✅
# 3. Default encoder bert-small (paper: 8L custom, ta: 4L pretrained) ✅
# 4. RMSNorm pre-norm (giống paper) ✅
# 5. Separate parameter groups cho LR riêng
#
# Architecture:
#   Input → Encoder(BERT) → E
#                             ├→ FeatureProjection(E) → E'
#                             └→ MambaDecoder(E) → H
#                                  └→ FeatureProjection(H) → H'
#                                       └→ CrossAttention(Q=H', K=E', V=E') → F
#                                            └→ Pooling → RMSNorm → Classifier → logits

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoConfig
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba_baseline import PureSSM


# ============================================================================
# RMSNorm (Gu & Dao, 2023 — dùng trong Mamba paper)
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Normalization — stable hơn LayerNorm cho deep models."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ============================================================================
# FEATURE PROJECTION MODULES (theo TransMamba paper)
# ============================================================================

class TransformerFeatureProjection(nn.Module):
    """
    Projection cho Transformer encoder features (theo paper).
    Linear → SiLU → Linear
    
    Giúp refine global features trước khi đưa vào cross-attention.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        return self.norm(x + self.proj(x))


class MambaFeatureProjection(nn.Module):
    """
    Projection cho Mamba decoder features (theo paper).
    Conv1x1 → SiLU → Conv1x1
    
    Conv1x1 tương đương pointwise projection, giữ nguyên sequence dimension.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, L, d_model) → transpose → conv → transpose back
        residual = x
        h = x.transpose(1, 2)  # (B, d_model, L)
        h = self.conv1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = h.transpose(1, 2)  # (B, L, d_model)
        return self.norm(self.dropout(h) + residual)


# ============================================================================
# FEATURE FUSION MODULE (đúng theo TransMamba paper)
# ============================================================================

class CrossAttentionFusionV2(nn.Module):
    """
    Feature Fusion — đúng theo TransMamba paper (Zhu et al., 2025).
    
    Pipeline:
    1. E' = TransformerFeatureProjection(encoder_output)    [Linear → SiLU → Linear]
    2. H' = MambaFeatureProjection(mamba_output)             [Conv1x1 → SiLU → Conv1x1]
    3. F  = CrossAttention(Q=H', K=E', V=E')                [Multi-head attention]
    4. Output = LayerNorm(H' + F)                            [Residual + norm]
    
    Paper ablation cho thấy: bỏ fusion → giảm performance đáng kể.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Step 1: Project encoder features
        self.encoder_proj = TransformerFeatureProjection(d_model, dropout)
        # Step 2: Project mamba features
        self.mamba_proj = MambaFeatureProjection(d_model, dropout)
        # Step 3: Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Step 4: Residual + norm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, mamba_output, encoder_output):
        # Project features (theo paper)
        E_proj = self.encoder_proj(encoder_output)  # (B, L, d)
        H_proj = self.mamba_proj(mamba_output)       # (B, L, d)
        
        # Cross-Attention: Q=MambaProj, K=V=EncoderProj
        attn_output, _ = self.cross_attn(
            query=H_proj,
            key=E_proj,
            value=E_proj,
        )
        
        # Residual + LayerNorm
        fused = self.norm(H_proj + self.dropout(attn_output))
        return fused


class CrossAttentionFusionSimple(nn.Module):
    """
    Simple Cross-Attention Fusion (không có feature projection).
    Giữ lại cho ablation comparison.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, mamba_output, encoder_output):
        attn_output, _ = self.cross_attn(
            query=mamba_output, key=encoder_output, value=encoder_output,
        )
        return self.norm(mamba_output + self.dropout(attn_output))


class AdditiveFusion(nn.Module):
    """Simple Additive Fusion: F = LayerNorm(H_mamba + E_encoder)."""
    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, mamba_output, encoder_output):
        return self.norm(self.dropout(mamba_output + encoder_output))


class NoFusion(nn.Module):
    """No Fusion: Chỉ dùng Mamba output."""
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, mamba_output, encoder_output):
        return self.norm(mamba_output)


# ============================================================================
# MAMBA DECODER STACK
# ============================================================================

class MambaDecoderStack(nn.Module):
    """
    Stack of PureSSM layers — sequential modeling component.
    Pre-norm architecture với residual connections.
    """
    def __init__(
        self, d_model: int, n_layers: int = 4,
        d_state: int = 16, d_conv: int = 4, expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PureSSM(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            RMSNorm(d_model) for _ in range(n_layers)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for ssm, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            residual = x
            x = norm(x)          # Pre-norm (more stable)
            x = ssm(x)
            x = dropout(x) + residual
        return x


# ============================================================================
# FUSION METHOD REGISTRY
# ============================================================================

FUSION_METHODS = {
    "cross_attention": CrossAttentionFusionV2,      # Full — theo paper (recommended)
    "cross_attention_simple": CrossAttentionFusionSimple,  # Simple — for ablation
    "additive": AdditiveFusion,
    "none": NoFusion,
}


# ============================================================================
# ENCODER PRESETS
# ============================================================================

ENCODER_PRESETS = {
    "bert-tiny": "prajjwal1/bert-tiny",           # 2L, 128d, 4.4M
    "bert-mini": "prajjwal1/bert-mini",            # 4L, 256d, 11.2M
    "bert-small": "prajjwal1/bert-small",          # 4L, 512d, 28.8M
    "bert-medium": "prajjwal1/bert-medium",        # 8L, 512d, 41.4M
    "bert-base": "bert-base-uncased",              # 12L, 768d, 110M
    "distilbert": "distilbert-base-uncased",       # 6L, 768d, 66M
}


# ============================================================================
# TRANSMAMBA CLASSIFIER
# ============================================================================

class TransMambaClassifier(nn.Module):
    """
    TransMamba-Cls: Hybrid Transformer + Mamba for Text Classification.
    
    Based on Zhu et al. (2025) "TransMamba" — improved version.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Input tokens                                                   │
    │       ↓                                                         │
    │  Pretrained Encoder (BERT-tiny/small/base)                      │
    │       │                                                         │
    │       ├──→ TransformerFeatureProjection ──→ E' (refined global)  │
    │       │    [Linear → SiLU → Linear]                             │
    │       │                                                         │
    │       └──→ MambaDecoderStack (N layers PureSSM)                 │
    │                 │                                               │
    │                 └──→ MambaFeatureProjection ──→ H' (refined seq) │
    │                      [Conv1x1 → SiLU → Conv1x1]                │
    │                                                                 │
    │  Cross-Attention: Q=H', K=E', V=E' ──→ F (fused)               │
    │       ↓                                                         │
    │  Mean Pooling → RMSNorm → Classifier → logits                   │
    └─────────────────────────────────────────────────────────────────┘
    
    Key features:
    - Feature projections trước cross-attention (theo paper)
    - Pre-norm (RMSNorm) trong decoder stack (stability)
    - Hỗ trợ multiple encoder sizes
    - Separate param groups cho LR riêng
    - Default 8 Mamba layers (aligned with paper: 16L)
    """
    
    def __init__(
        self,
        encoder_name: str = "prajjwal1/bert-small",  # Gần paper (4L vs paper 8L)
        n_mamba_layers: int = 8,           # Gần paper (8L vs paper 16L)
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_labels: int = 2,
        dropout: float = 0.1,
        fusion: str = "cross_attention",
        num_heads_fusion: int = 4,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        # Resolve encoder preset
        if encoder_name in ENCODER_PRESETS:
            encoder_name = ENCODER_PRESETS[encoder_name]
        
        self.encoder_name = encoder_name
        self.fusion_type = fusion
        self.num_labels = num_labels
        
        # ── Component 1: Transformer Encoder (pretrained) ──
        self.encoder = AutoModel.from_pretrained(encoder_name)
        d_model = self.encoder.config.hidden_size
        self.d_model = d_model
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # ── Component 2: Mamba Decoder (PureSSM, train from scratch) ──
        self.mamba_decoder = MambaDecoderStack(
            d_model=d_model,
            n_layers=n_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # ── Component 3: Feature Fusion ──
        if fusion not in FUSION_METHODS:
            raise ValueError(f"Unknown fusion: {fusion}. Choose from {list(FUSION_METHODS.keys())}")
        
        fusion_cls = FUSION_METHODS[fusion]
        if fusion in ("cross_attention", "cross_attention_simple"):
            self.fusion_module = fusion_cls(d_model, num_heads=num_heads_fusion, dropout=dropout)
        else:
            self.fusion_module = fusion_cls(d_model, dropout=dropout)
        
        # ── Component 4: Final Norm + Classifier Head ──
        self.final_norm = RMSNorm(d_model)   # RMSNorm trước classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Step 1: Transformer Encoder → global features
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # (B, L, d_model)
        
        # Step 2: Mamba Decoder → sequential features
        mamba_output = self.mamba_decoder(encoder_output)  # (B, L, d_model)
        
        # Step 3: Feature Fusion → combined
        fused = self.fusion_module(mamba_output, encoder_output)  # (B, L, d_model)
        
        # Step 4: Pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (fused * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = fused.mean(dim=1)
        
        # Step 5: Final Norm + Classification
        pooled = self.final_norm(pooled)
        logits = self.classifier(pooled)
        
        # Loss
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {"loss": loss, "logits": logits}
    
    def get_param_groups(self, encoder_lr=5e-4, decoder_lr=1e-3):
        """Separate parameter groups for different learning rates (theo paper)."""
        encoder_params = list(self.encoder.parameters())
        decoder_params = (
            list(self.mamba_decoder.parameters()) +
            list(self.fusion_module.parameters()) +
            list(self.final_norm.parameters()) +
            list(self.classifier.parameters())
        )
        return [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ]
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        mamba_params = sum(p.numel() for p in self.mamba_decoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())
        classifier_params = (
            sum(p.numel() for p in self.final_norm.parameters()) +
            sum(p.numel() for p in self.classifier.parameters())
        )
        
        return {
            "encoder_name": self.encoder_name,
            "d_model": self.d_model,
            "fusion_type": self.fusion_type,
            "total_params": self.count_parameters(),
            "encoder_params": encoder_params,
            "mamba_decoder_params": mamba_params,
            "fusion_params": fusion_params,
            "classifier_params": classifier_params,
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing TransMamba-Cls (Aligned with Paper)")
    print("=" * 60)
    
    # Test all fusion types with bert-tiny (fast, for validation)
    for fusion in ["cross_attention", "cross_attention_simple", "additive", "none"]:
        print(f"\n--- Fusion: {fusion} ---")
        
        model = TransMambaClassifier(
            encoder_name="prajjwal1/bert-tiny",  # Fast test
            n_mamba_layers=8,                      # 8L decoder (aligned with paper)
            num_labels=2,
            fusion=fusion,
        )
        
        info = model.get_model_info()
        print(f"  Total params: {info['total_params']:,}")
        print(f"  Encoder: {info['encoder_params']:,}")
        print(f"  Mamba Decoder: {info['mamba_decoder_params']:,}")
        print(f"  Fusion: {info['fusion_params']:,}")
        print(f"  Classifier: {info['classifier_params']:,}")
        
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.randint(0, 2, (batch_size,))
        
        output = model(input_ids, attention_mask, labels)
        print(f"  Logits shape: {output['logits'].shape}")
        print(f"  Loss: {output['loss'].item():.4f}")
    
    # Test param groups
    print(f"\n--- Param Groups (bert-tiny, 8L) ---")
    model = TransMambaClassifier(encoder_name="prajjwal1/bert-tiny", n_mamba_layers=8)
    groups = model.get_param_groups(encoder_lr=5e-4, decoder_lr=1e-3)
    for i, g in enumerate(groups):
        n_params = sum(p.numel() for p in g["params"])
        print(f"  Group {i}: {n_params:,} params, lr={g['lr']}")
    
    # Test encoder presets (2 encoder sizes used in experiments)
    print(f"\n--- Encoder Presets (Encoder Scaling) ---")
    for name in ["bert-tiny", "bert-small"]:
        model = TransMambaClassifier(encoder_name=name, n_mamba_layers=8)
        print(f"  {name}: d_model={model.d_model}, total={model.count_parameters():,}")
    
    print(f"\n{'=' * 60}")
    print("✅ All tests passed! (TransMamba-Cls — 2 encoder sizes)")
    print("=" * 60)


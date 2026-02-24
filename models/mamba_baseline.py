# Pure PyTorch SSM Implementation — MAMBA BASELINE
# Copy from bimamba_project — dùng làm baseline và decoder cho TransMamba
# Không cần mamba-ssm, chạy được mọi nơi!

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PureSSM(nn.Module):
    """
    Pure PyTorch implementation of Selective State Space Model (SSM).
    Không cần mamba-ssm CUDA extensions.
    
    Tham khảo: Mamba paper (Gu & Dao, 2023)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        bias: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank
        
        # Input projection: x -> (z, x)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt_proj bias
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # A parameter (log scale for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seq_len]  # Trim padding
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM
        y = self.ssm(x)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x):
        batch, seq_len, d_inner = x.shape
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()
        
        # Project x to get dt, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Compute delta (time step)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        
        # Selective scan
        y = self.selective_scan(x, dt, A, B, C, D)
        
        return y
    
    def selective_scan(self, x, dt, A, B, C, D):
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, d_inner)
            dt_t = dt[:, t, :]  # (B, d_inner)
            B_t = B[:, t, :]  # (B, d_state)
            C_t = C[:, t, :]  # (B, d_state)
            
            # Discretize
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)
            
            # State update: h = dA * h + dB * x
            h = dA * h + dB * x_t.unsqueeze(-1)
            
            # Output: y = C * h + D * x
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + D * x_t  # (B, d_inner)
            
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return y


class PureMambaClassifier(nn.Module):
    """
    Pure Mamba Classifier — BASELINE model.
    Dùng để so sánh với TransMamba-Cls.
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_labels: int = 2,
        dropout: float = 0.1,
        max_length: int = 128,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            PureSSM(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.pos_embedding(position_ids)
        x = self.embed_dropout(x)
        
        for mamba, norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba(x)
            x = norm(x + residual)
        
        x = self.final_norm(x)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)
        
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {"loss": loss, "logits": logits}
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test
if __name__ == "__main__":
    print("Testing PureMambaClassifier (Baseline)...")
    
    model = PureMambaClassifier(vocab_size=30522, d_model=256, n_layers=4, num_labels=2)
    print(f"Total parameters: {model.count_parameters():,}")
    
    input_ids = torch.randint(0, 30522, (2, 64))
    labels = torch.randint(0, 2, (2,))
    output = model(input_ids, labels=labels)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print("✅ Baseline test passed!")

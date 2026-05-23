"""
model.py
========
Mô hình ISLR nhẹ cho nhận diện ngôn ngữ ký hiệu đơn lẻ.

Kiến trúc: KeypointLSTM
  Input : (Batch, T, 225)  — T=30 frames, 75 keypoints × 3 coords
  → Linear projection (225 → 256)
  → BatchNorm1d
  → Bidirectional LSTM (256 hidden, 2 layers, dropout=0.3)
  → Lấy hidden state cuối → Linear (512 → num_classes)

Tại sao Bi-LSTM thay vì Transformer?
  - Dataset nhỏ (~15-30 samples/class) → Transformer dễ overfit.
  - Bi-LSTM nhẹ, huấn luyện nhanh, đủ tốt để nắm bắt temporal dynamics.
"""

import torch
import torch.nn as nn


class KeypointLSTM(nn.Module):
    """
    Mạng LSTM nhẹ cho ISLR từ MediaPipe keypoints.

    Args:
        num_classes : số lớp nhận diện (số gloss)
        input_dim   : chiều vector đầu vào mỗi frame (= 75 * 3 = 225)
        hidden_dim  : số unit ẩn LSTM
        num_layers  : số lớp LSTM
        dropout     : tỉ lệ dropout (áp dụng giữa các lớp LSTM)
        bidirectional: dùng Bi-LSTM hay không
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 237,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Projection đầu vào
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_directions

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif isinstance(param, nn.Linear):
                nn.init.kaiming_normal_(param.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)  →  logits: (B, num_classes)
        """
        B, T, D = x.shape

        # Input projection — BatchNorm1d yêu cầu (B, C) hoặc (B, C, L)
        x = x.reshape(B * T, D)
        x = self.input_proj(x)           # (B*T, hidden_dim)
        x = x.reshape(B, T, -1)          # (B, T, hidden_dim)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)  # h_n: (num_layers * num_dir, B, hidden_dim)

        # Lấy hidden state của lớp cuối cùng, cả 2 chiều
        # h_n[-1] = forward cuối; h_n[-2] = backward cuối (nếu bidirectional)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2*hidden_dim)
        else:
            h_last = h_n[-1]                                  # (B, hidden_dim)

        logits = self.classifier(h_last)  # (B, num_classes)
        return logits


# ═══════════════════════════════════════════════════════════
# Biến thể với Self-Attention pooling (tùy chọn)
# ═══════════════════════════════════════════════════════════
class KeypointTransformer(nn.Module):
    """
    Phiên bản nhỏ dùng Transformer Encoder (dự phòng khi dataset lớn hơn).
    Hiện tại khuyến nghị dùng KeypointLSTM.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 237,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_seq_len: int = 60,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Learnable positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = self.input_proj(x) + self.pos_embed(pos)         # (B, T, model_dim)
        x = self.transformer(x)                               # (B, T, model_dim)
        x = x.mean(dim=1)                                     # Global average pooling
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════
# Test model nhanh
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    num_classes = 3   # TOI, BAN, THCH
    model = KeypointLSTM(num_classes=num_classes)
    print(model)

    # Đếm tham số
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTổng tham số: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Forward pass test
    dummy = torch.randn(4, 30, 225)   # batch=4, T=30, feature=225
    out   = model(dummy)
    print(f"Output shape: {out.shape}")  # (4, 3)

"""
model.py
========
Mô hình ISLR nhẹ cho nhận diện ngôn ngữ ký hiệu đơn lẻ.

Kiến trúc có sẵn:
  1. KeypointLSTM         : Bi-LSTM lấy hidden state frame cuối → Classifier
  2. KeypointLSTMAttention: Bi-LSTM + Attention Pooling — mô hình học cách
     đánh trọng số cho từng frame, tập trung vào đỉnh chuyển động quan trọng nhất.
     Phù hợp với dataset nhỏ (20-50 mẫu/class), thường tốt hơn LSTM thuần.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# Model 1: KeypointLSTM (baseline)
# ═══════════════════════════════════════════════════════════
class KeypointLSTM(nn.Module):
    """
    Mạng Bi-LSTM nhẹ cho ISLR từ MediaPipe keypoints.
    Lấy hidden state của frame cuối cùng để phân loại.

    Args:
        num_classes  : số lớp nhận diện (số gloss)
        input_dim    : chiều vector đầu vào mỗi frame (= 75 * 3 = 225)
        hidden_dim   : số unit ẩn LSTM
        num_layers   : số lớp LSTM
        dropout      : tỉ lệ dropout
        bidirectional: dùng Bi-LSTM hay không
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 225,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_directions
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = x.reshape(B * T, D)
        x = self.input_proj(x)
        x = x.reshape(B, T, -1)

        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        return self.classifier(h_last)


# ═══════════════════════════════════════════════════════════
# Model 2: KeypointLSTMAttention (khuyến nghị)
# ═══════════════════════════════════════════════════════════
class KeypointLSTMAttention(nn.Module):
    """
    Bi-LSTM + Attention Pooling cho ISLR từ MediaPipe keypoints.

    Cơ chế hoạt động:
      1. Input → Linear Projection → Bi-LSTM → chuỗi hidden states (B, T, D)
      2. Attention Head: học một "câu hỏi" để hỏi từng frame
         "Frame này có quan trọng không?" → cho ra điểm số (score)
      3. Softmax trên T frames → Trọng số Attention (tổng = 1)
      4. Weighted Sum toàn bộ chuỗi → 1 vector đại diện cho cả câu (B, D)
      5. Classifier → logits

    Lợi ích so với KeypointLSTM:
      - Trong Sign Language, đỉnh chuyển động (peak) thường nằm ở GIỮA chuỗi,
        không phải cuối. Attention tự học tìm ra đỉnh đó mà không cần hướng dẫn.
      - Chỉ thêm ~33K tham số so với LSTM thuần → Không bị overfit trên data nhỏ.

    Args:
        num_classes  : số lớp nhận diện
        input_dim    : chiều đầu vào mỗi frame (= 75 * 3 = 225)
        hidden_dim   : số unit ẩn LSTM
        num_layers   : số lớp LSTM
        dropout      : tỉ lệ dropout
        bidirectional: dùng Bi-LSTM hay không
        attn_dim     : chiều nội bộ của Attention Head (128 là đủ)
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 225,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1
        lstm_out_dim        = hidden_dim * self.num_directions

        # Projection đầu vào
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # ── Attention Head ──
        # Linear(D→attn_dim) → Tanh → Linear(attn_dim→1) → scalar mỗi frame
        self.attn_fc = nn.Sequential(
            nn.Linear(lstm_out_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)  →  logits: (B, num_classes)
        """
        B, T, D = x.shape

        # Input projection
        x = x.reshape(B * T, D)
        x = self.input_proj(x)      # (B*T, hidden_dim)
        x = x.reshape(B, T, -1)     # (B, T, hidden_dim)

        # Bi-LSTM: lấy TOÀN BỘ output mỗi frame
        lstm_out, _ = self.lstm(x)  # (B, T, lstm_out_dim)

        # ── Attention Pooling ──
        attn_scores  = self.attn_fc(lstm_out)             # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)      # (B, T, 1)
        context      = (attn_weights * lstm_out).sum(dim=1)  # (B, lstm_out_dim)

        return self.classifier(context)                   # (B, num_classes)


# ═══════════════════════════════════════════════════════════
# Test nhanh
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    B, T, D, C = 4, 15, 225, 6

    for name, model in [
        ("KeypointLSTM", KeypointLSTM(C, D)),
        ("KeypointLSTMAttention", KeypointLSTMAttention(C, D)),
    ]:
        out = model(torch.randn(B, T, D))
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:30s} | output={out.shape} | params={params:,}")

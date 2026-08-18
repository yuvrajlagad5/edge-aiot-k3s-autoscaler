import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionLSTM(nn.Module):
    """
    Temporal-Attention LSTM (TA-LSTM) Predictor for workload forecasting.
    Implements Equations (1)-(11) from the paper.
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
        super(TemporalAttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Recurrent Core (Equations 1-6)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Temporal Attention Mechanism Parameters (Equations 7 & 10)
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.va = nn.Linear(hidden_dim, 1, bias=False)
        
        self.Ws = nn.Linear(hidden_dim * 2, hidden_dim)
        self.Wy = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for time-series feature sequence.
        Input x shape: (batch_size, seq_len, input_dim) -> [C_t, M_t, R_t]
        """
        # lstm_out: (B, T, dh), h_n: (1, B, dh)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        h_T = h_n.squeeze(0).unsqueeze(1)  # Shape: (B, 1, dh)
        
        # Calculate alignment scores e_t (Equation 7)
        # e_t = v_a^T * tanh(W_a * h_t + U_a * h_T)
        scores = self.va(torch.tanh(self.Wa(lstm_out) + self.Ua(h_T)))  # (B, T, 1)
        
        # Normalized scalar attention weights alpha_t via softmax (Equation 8)
        alpha = F.softmax(scores, dim=1)  # Shape: (B, T, 1)
        
        # Dynamic context vector c_att (Equation 9)
        c_att = torch.sum(alpha * lstm_out, dim=1)  # Shape: (B, dh)
        
        # Context-aware representation h_tilde_T (Equation 10)
        h_tilde = torch.tanh(self.Ws(torch.cat((c_att, h_n.squeeze(0)), dim=1)))
        
        # Forecasted demand y_hat_{t+delta_t} (Equation 11)
        y_hat = self.Wy(h_tilde)
        
        return y_hat, alpha

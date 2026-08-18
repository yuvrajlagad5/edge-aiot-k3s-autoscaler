"""
Temporal-Attention LSTM (TA-LSTM) Autoscaler Package
---------------------------------------------------
Provides predictive workload forecasting and dynamic Kubernetes 
pod autoscaling for Edge-AIoT environments.
"""

from .ta_lstm import TemporalAttentionLSTM
from .controller import AdaptiveAutoscaler

__all__ = ["TemporalAttentionLSTM", "AdaptiveAutoscaler"]

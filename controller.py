import math
import time
import numpy as np
import torch
from collections import deque
from kubernetes import client, config
from ta_lstm import TemporalAttentionLSTM

class AdaptiveAutoscaler:
    """
    Uncertainty-Aware Scaling Engine (Equations 12-13).
    Monitors telemetry, invokes TA-LSTM, and patches Kubernetes Deployment replicas.
    """
    def __init__(self, model_path, namespace="default", deployment_name="aiot-service"):
        # Initialize Kubernetes API Client
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
            
        self.apps_v1 = client.AppsV1Api()
        self.namespace = namespace
        self.deployment_name = deployment_name
        
        # Load pre-trained TA-LSTM Predictor
        self.model = TemporalAttentionLSTM(input_dim=3, hidden_dim=64, output_dim=1)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        # Autoscaling Parameters
        self.R_target = 50.0      # Nominal pod request capacity (req/sec)[cite: 6]
        self.N_min = 1            # Minimum pod replicas[cite: 6]
        self.N_max = 10           # Maximum pod replicas[cite: 6]
        self.beta = 1.5           # Adaptive headroom multiplier[cite: 6]
        self.errors = deque(maxlen=50) # Running error lookback window K=50[cite: 6]

    def compute_target_replicas(self, feature_sequence, actual_demand=None):
        """
        Computes desired replica count incorporating adaptive variance headroom gamma.
        feature_sequence: numpy array of shape (T, 3) representing [C_t, M_t, R_t][cite: 6]
        """
        input_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)  # Shape: (1, T, 3)[cite: 6]
        
        with torch.no_grad():
            y_hat, _ = self.model(input_tensor)
            predicted_demand = max(0.0, y_hat.item())
            
        # Update running prediction error variance (sigma_err^2)
        if actual_demand is not None:
            self.errors.append(actual_demand - predicted_demand)
            
        var_err = np.var(self.errors) if len(self.errors) >= 5 else 0.0
        
        # Calculate adaptive uncertainty headroom gamma (Equation 12)
        gamma = self.beta * var_err[cite: 6]
        
        # Calculate N_desired (Equation 13)
        raw_replicas = (predicted_demand * (1.0 + gamma)) / self.R_target[cite: 6]
        N_desired = min(self.N_max, max(self.N_min, math.ceil(raw_replicas)))[cite: 6]
        
        return N_desired, predicted_demand, gamma

    def patch_k8s_replicas(self, target_replicas):
        """Patches the K3s target deployment scale."""
        body = {"spec": {"replicas": target_replicas}}
        self.apps_v1.patch_namespaced_deployment_scale(
            name=self.deployment_name,
            namespace=self.namespace,
            body=body
        )
        print(f"[Autoscaler] Patched deployment '{self.deployment_name}' to {target_replicas} replicas.")

if __name__ == "__main__":
    autoscaler = AdaptiveAutoscaler(model_path="ta_lstm_model.pth")
    print("Autoscaler Controller Initialized successfully.")

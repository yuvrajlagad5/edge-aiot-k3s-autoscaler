import pandas as pd
import numpy as np

def compute_benchmark_metrics(csv_file_path, slo_threshold_ms=100.0):
    """
    Calculates median (p50), tail latency (p95), SLO violation rate, 
    and HTTP 5xx error rate from test execution logs.
    """
    df = pd.read_csv(csv_file_path)
    
    # Calculate Latency Percentiles
    p50_latency = np.percentile(df['response_time_ms'], 50)
    p95_latency = np.percentile(df['response_time_ms'], 95)
    
    # Calculate SLO Violation Rate (> 100ms)
    slo_violations = (df['response_time_ms'] > slo_threshold_ms).sum()
    slo_violation_rate = (slo_violations / len(df)) * 100.0
    
    # Calculate HTTP 5xx Error Rate
    five_xx_errors = (df['status_code'] >= 500).sum()
    error_rate = (five_xx_errors / len(df)) * 100.0
    
    print("=" * 50)
    print("           TA-LSTM BENCHMARK RESULTS            ")
    print("=" * 50)
    print(f"Average Latency (p50)      : {p50_latency:.2f} ms")
    print(f"Tail Latency (p95)         : {p95_latency:.2f} ms")
    print(f"SLO Violation Rate (>100ms): {slo_violation_rate:.2f}%")
    print(f"HTTP 5xx Error Rate        : {error_rate:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    # Example usage: compute_benchmark_metrics("locust_stats.csv")
    pass

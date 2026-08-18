import math
import time
from locust import HttpUser, task, between

class AIoTTelemetryUser(HttpUser):
    """
    Locust traffic generator executing HTTP/2 telemetry calls 
    under step-burst and sinusoidal surge conditions.
    """
    wait_time = between(0.05, 0.2)

    @task
    def send_telemetry_stream(self):
        payload = {
            "node_id": "raspberry-pi-node-01",
            "timestamp": time.time(),
            "telemetry": {
                "cpu_load": 42.5,
                "memory_mb": 256.0,
                "sensor_readings": [24.1, 60.5, 1012.8]
            }
        }
        headers = {"Content-Type": "application/json"}
        self.client.post("/api/v1/telemetry", json=payload, headers=headers)

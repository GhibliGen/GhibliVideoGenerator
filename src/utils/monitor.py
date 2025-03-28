import time
import torch
from typing import Dict

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = {}
        
    def start(self) -> None:
        """Start monitoring performance"""
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return metrics"""
        if self.start_time is None:
            raise ValueError("Monitoring not started")
            
        elapsed = time.time() - self.start_time
        self.metrics = {
            "processing_time": elapsed,
            "peak_memory_mb": (torch.cuda.max_memory_allocated() / 1024**2 
                             if torch.cuda.is_available() else 0),
            "avg_fps": 24 / elapsed  # Assuming 24 frames target
        }
        self.start_time = None
        return self.metrics

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RooflineEstimate:
    name: str
    bytes_moved: int
    flops: int

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes_moved

    def achieved_gbps(self, latency_ms: float) -> float:
        return self.bytes_moved / latency_ms / 1e6

    def achieved_gflops(self, latency_ms: float) -> float:
        return self.flops / latency_ms / 1e6

    def achieved_tflops(self, latency_ms: float) -> float:
        return self.flops / latency_ms / 1e9


def total_bytes(element_size: int, *, reads: int, writes: int) -> int:
    return element_size * (reads + writes)

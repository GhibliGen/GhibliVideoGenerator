import pytest
import torch
from src.models.style_network import GhibliStyleNet
from src.temporal.temporal_coherence import TemporalCoherenceModule
from src.interpolation.frame_interpolator import AdaptiveFrameInterpolator
from src.utils.preprocessor import ImagePreprocessor

@pytest.fixture
def config():
    return {
        "model": {"layers": 4, "channels": 16},
        "temporal": {"window_size": 3, "consistency_weight": 0.5},
        "interpolation": {"method": "adaptive", "complexity": 2},
        "processing": {"resolution": [256, 256]}
    }

def test_style_network(config):
    model = GhibliStyleNet(config["model"])
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model.generate(input_tensor)
    assert output.shape == input_tensor.shape
    assert not torch.isnan(output).any()

def test_temporal_coherence(config):
    module = TemporalCoherenceModule(config["temporal"])
    frames = torch.randn(1, 5, 3, 256, 256)
    coherent = module.enforce_coherence(frames)
    assert coherent.shape == frames.shape
    assert torch.all(coherent >= -1) and torch.all(coherent <= 1)

def test_frame_interpolator(config):
    interpolator = AdaptiveFrameInterpolator(config["interpolation"])
    frames = torch.randn(1, 5, 3, 256, 256)
    interpolated = interpolator.interpolate(frames, target_fps=10, duration=2.0)
    assert interpolated.shape[1] == 20  # 10 fps * 2 seconds
    assert interpolated.shape[2:] == frames.shape[2:]

def test_preprocessor(config):
    preprocessor = ImagePreprocessor(config["processing"])
    # Note: Would need a real image for full testing
    with pytest.raises(ValueError):
        preprocessor.process("nonexistent.jpg")

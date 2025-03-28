import torch
import yaml
import logging
from src.models.style_network import GhibliStyleNet
from src.temporal.temporal_coherence import TemporalCoherenceModule
from src.interpolation.frame_interpolator import AdaptiveFrameInterpolator
from src.utils.preprocessor import ImagePreprocessor
from src.utils.postprocessor import VideoPostprocessor
from src.utils.monitor import PerformanceMonitor
import argparse
import os
from typing import Dict

class GhibliGenerator:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_components()
        self._setup_logging()
        self.monitor = PerformanceMonitor()
        
    def _load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_components(self) -> None:
        self.style_net = GhibliStyleNet(self.config["model"]).to(self.device)
        self.temporal_module = TemporalCoherenceModule(self.config["temporal"])
        self.frame_interpolator = AdaptiveFrameInterpolator(self.config["interpolation"])
        self.preprocessor = ImagePreprocessor(self.config["processing"])
        self.postprocessor = VideoPostprocessor()
        self._load_weights()
        
    def _load_weights(self) -> None:
        checkpoint = torch.load(self.config["model"]["checkpoint_path"])
        self.style_net.load_state_dict(checkpoint["model_state"])
        logging.info(f"Loaded weights from {self.config['model']['checkpoint_path']}")
        
    def _setup_logging(self) -> None:
        os.makedirs(self.config["logging"]["output_dir"], exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config['logging']['output_dir']}/app.log"),
                logging.StreamHandler()
            ]
        )
        
    @torch.no_grad()
    def process_image_to_video(self, input_path: str, output_path: str) -> None:
        try:
            self.monitor.start()
            
            input_tensor = self.preprocessor.process(input_path)
            styled_frames = self.style_net.generate(input_tensor)
            coherent_frames = self.temporal_module.enforce_coherence(styled_frames)
            final_frames = self.frame_interpolator.interpolate(
                coherent_frames, 
                self.config["interpolation"]["target_fps"],
                duration=10.0
            )
            self.postprocessor.save_video(final_frames, output_path)
            
            metrics = self.monitor.stop()
            logging.info(f"Processing completed. Metrics: {metrics}")
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}", exc_info=True)
            raise
            
def main():
    parser = argparse.ArgumentParser(description="Ghibli Generator Production System")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output video path")
    
    args = parser.parse_args()
    
    generator = GhibliGenerator(args.config)
    generator.process_image_to_video(args.input, args.output)

if __name__ == "__main__":
    main()

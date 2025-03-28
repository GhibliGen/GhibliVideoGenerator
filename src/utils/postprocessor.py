import torch
import torchvision
import numpy as np
import ffmpeg

class VideoPostprocessor:
    def __init__(self):
        self.denormalize = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
    def save_video(self, frames: torch.Tensor, 
                  output_path: str, 
                  fps: int = 24) -> None:
        """Save processed frames as video"""
        # Denormalize frames
        frames = self.denormalize(frames)
        frames = torch.clamp(frames, 0, 1)
        
        # Convert to numpy
        frames_np = frames.cpu().numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        
        try:
            # Use FFmpeg to create video
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', 
                      pix_fmt='rgb24', 
                      s=f'{frames.shape[-1]}x{frames.shape[-2]}')
                .output(output_path, pix_fmt='yuv420p', 
                       vcodec='libx264', 
                       r=fps)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            
            for frame in frames_np.transpose(1, 2, 3, 0):
                process.stdin.write(frame.tobytes())
                
            process.stdin.close()
            process.wait()
            
        except ffmpeg.Error as e:
            raise RuntimeError(f"Video creation failed: {str(e)}")

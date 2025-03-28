<h1 align="center">Ghibli Generator - AI-Powered Animation System</h1>

<p align="center">
  <img src="Graphic.png" alt="Ghibli Generator Logo" width="300">
</p>

<p align="center">
  <a href="https://ghiblianimator.tech/">🌐 Visit Our Website</a>
   <a href="https://x.com/GhibliVids">🐦 Follow Us on Twitter</a>
</p>

Enterprise-grade AI system for transforming images into Studio Ghibli-style animated videos.

## 🚀 Features
- **Production-ready** with comprehensive testing
- **GPU-accelerated** processing pipeline for efficiency
- **Configurable parameters** via YAML for fine-tuned control
- **Extensive logging & monitoring** for performance tracking
- **Docker deployment support** for seamless containerization

---

## 📌 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ghibli-generator.git
cd ghibli-generator
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Configuration
```bash
cp config/example_config.yaml config/config.yaml
```

---

## 🛠 Usage
### Convert an Image to an Animated Video
```bash
python main.py --config config/config.yaml --input image.jpg --output video.mp4
```

### Run Tests
```bash
pytest tests/
```

---

## 📂 Project Structure
```
📦 ghibli-generator
├── 📂 config            # Configuration files
│   ├── config.yaml      # Main config file
│   ├── example_config.yaml  # Example config template
├── 📂 models            # Model architectures & weights
├── 📂 scripts           # Helper scripts for training & inference
├── 📂 tests             # Unit tests
├── main.py              # Entry point for animation generation
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 📑 Configuration Example (`config/example_config.yaml`)
```yaml
model:
  layers: 32
  channels: 64
  checkpoint_path: "weights/ghibli_style.pth"

temporal:
  window_size: 5
  consistency_weight: 0.8
  flow_method: "horn_schunck"

interpolation:
  method: "adaptive"
  complexity: 4
  target_fps: 24

processing:
  batch_size: 4
  resolution: [512, 512]
  num_workers: 4

logging:
  level: "INFO"
  output_dir: "logs"
```

---

## 📜 Requirements (`requirements.txt`)
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.5.0
pyyaml>=6.0
pytest>=7.4.0
ffmpeg-python>=0.2.0
opencv-python>=4.8.0
```

---

## 📌 Deployment with Docker
### Build & Run the Container
```bash
docker build -t ghibli-generator .
docker run --rm -v $(pwd):/app ghibli-generator --config config/config.yaml --input image.jpg --output video.mp4
```

---

## 🎨 Example Output
> A side-by-side comparison of input images and the AI-generated Ghibli-style animations.

---

## 🏗 Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

---

## 📄 License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## ✨ Acknowledgments
Inspired by the beautiful animation style of Studio Ghibli.

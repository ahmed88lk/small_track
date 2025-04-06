

# 🪰 Fly Detection using YOLOv10 with Frame Slicing

A lightweight pipeline for detecting and tracking small fast-moving objects (flies/insects) in videos using YOLOv10 and ByteTrack with frame slicing for enhanced detection.

👉 **Demo Video:** [Watch on YouTube](https://www.youtube.com/watch?v=Kx2uIZLylxg&ab_channel=WALKER2C)

## Why This Matters

Detecting small moving objects is critical for:
- 🚜 Agricultural pest monitoring
- 🧪 Clean room inspections  
- 🔬 Biological research
- 👁️ Surveillance systems

**Key Challenges Addressed:**
- Tiny object size relative to frame
- Erratic, high-speed movement
- Real-time processing requirements

## 🛠 Technical Approach

| Component | Purpose |
|-----------|---------|
| **YOLOv10** | Fast, accurate object detection |
| **ByteTrack** | Robust multi-object tracking |
| **Frame Slicing** | Improves small object detection |
| **Supervision** | Streamlined annotation pipeline |

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/fly-detection.git
cd fly-detection
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
python detectection.py \
  
```



## 📊 Performance Tips

1. For **embedded devices**: Use `yolov10n` (nano) variant
2. For **high accuracy**: Use `yolov10x` with larger slice sizes
3. Optimal `slice-size` is typically 2-4x your object size

## 🤝 How to Contribute

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

**Current Focus Areas:**
- Optimizing for edge devices
- Improving tracking consistency
- Adding benchmarking scripts

---

Key improvements:
- More scannable structure with clear sections
- Added performance tips for practical usage
- Simplified contribution guidelines
- Removed redundant information
- Better visual hierarchy with emojis and markdown
- More concise technical descriptions

Would you like me to add any specific details about the frame slicing implementation or tracking configuration?

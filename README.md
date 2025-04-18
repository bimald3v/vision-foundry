# Vision Foundry

A computer vision system for warehouse object detection using YOLOv8. This system can detect and classify various warehouse-related objects in both images and videos, with custom mappings for warehouse-specific terminology.

## Features

- Real-time object detection using YOLOv8
- Custom warehouse-specific label mapping (e.g., 'car' → 'forklift')
- Configurable confidence threshold for detections
- Support for video processing with visual annotations
- Color-coded object categories for easy visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bimald3v/vision-foundry.git
cd vision-foundry
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from warehouse_detector import WarehouseObjectDetector

# Initialize detector with custom confidence threshold (default: 0.3)
detector = WarehouseObjectDetector(confidence_threshold=0.4)

# Process a video file
detector.detect_objects_in_video(
    video_path='input_video.mp4',
    output_path='output_video.mp4'
)
```

## Class Mapping

The system maps standard COCO dataset classes to warehouse-specific labels:

- person → warehouse worker
- truck/car → forklift
- backpack/handbag → warehouse equipment
- suitcase/box/laptop → cardboard box
- tv → pallet rack

## Color Coding

Detections are color-coded for easy identification:
- Warehouse workers: Blue
- Forklifts: Yellow
- Cardboard boxes: Brown
- Warehouse equipment: Green
- Pallet racks: Red
- Unknown objects: Gray

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- Ultralytics YOLOv8

See `requirements.txt` for specific version requirements.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Object Detection with Flutter (TensorFlow Lite)

This Flutter application demonstrates **real-time human detection** using a pre-trained **TensorFlow Lite object detection model**. It allows you to pick an image from your gallery, runs inference using the `detect.tflite` model, and draws bounding boxes around detected people.

## Features

- Load and use a TFLite model (`detect.tflite`) in Flutter.
- Use `image_picker` to select images from the gallery.
- Detect humans (class 0 in COCO dataset).
- Draw bounding boxes for detected humans.

## Requirements

- Flutter 3.x
- Android SDK + Emulator or Physical Device
- Android NDK 27.0.12077973 (set via `ndkVersion` in `build.gradle.kts`)
- TFLite model `detect.tflite` (must be placed in `assets/`)

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/object-detection-poc.git
cd object-detection-poc
```

### 2. Add the TFLite model
Place your TFLite model in the assets/ directory:

```bash
/assets/models/detect.tflite
Update your pubspec.yaml to include:
```
```yaml
Copy
Edit
flutter:
  assets:
    - assets/detect.tflite
```


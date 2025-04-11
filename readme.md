# Orbbec Gemini 2 Toolkit

A Python toolkit for working with Orbbec Gemini 2 depth cameras. This toolkit provides high-performance access to color and depth streams with multi-threaded frame acquisition. Documentation is available at [Overleaf](https://www.overleaf.com/read/fjsvqhqkqmcc#86f71c).

## Features

- Multi-threaded frame acquisition for optimal performance
- Complete camera configuration (depth modes, profiles, etc.)
- Hardware-aligned RGB and depth streams
- Depth data post-processing with configurable filters
- Real-time visualization utilities
- Thread-safe frame buffer management

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- pyorbbecsdk

## Installation

```bash
# Clone the repository
git clone https://github.com/rusuanjun007/OrbbecGemini2Toolkit.git
```

## Quick Start

```python
import cv2
from gemini2 import Gemini2

# Create and initialize the camera
camera = Gemini2()
camera.init_camera()  # Uses default settings

try:
    # Main loop
    while True:
        # Get the latest frame (non-blocking)
        frame_data = camera.get_latest_frame()
        
        if frame_data is not None:
            color_image = frame_data["color_image"]
            depth_data = frame_data["depth_data"]
            
            # Display the frames
            camera.visualise_frame(color_image, depth_data)
            
            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break
                
finally:
    # Ensure proper cleanup
    camera.stop_acquisition_thread()
    camera.stop_pipeline()
    cv2.destroyAllWindows()
```

## Performance Monitoring

The example script demonstrates how to calculate and display the frames per second (FPS):

```python
import time

start_time = time.time()
frames_processed = 0

while True:
    # Process frames...
    frames_processed += 1
    
    # Calculate FPS every 100 frames
    if frames_processed % 100 == 0:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"FPS: {100 / elapsed:.2f}")
        start_time = time.time()
```

## Camera Configuration

### Depth Work Modes

```python
# Available depth work modes
camera.get_depth_work_mode()  # Print all available modes

# Set depth work mode (0-4)
# 0: Unbinned Dense Default
# 1: Binned Sparse Default
# 2: Unbinned Sparse Default
# 3: In-scene Calibration
# 4: Obstacle Avoidance
camera.set_depth_work_mode(0)
```

### Stream Profiles

```python
# List available depth and color profiles
camera.get_depth_stream_profile()
camera.get_color_steam_profile()

# Set profiles by index
camera.set_depth_stream_profile(2)  # 1280x800@30 Y16
camera.set_color_stream_profile(1)  # 1920x1080@30 RGB
```

### Accessing Frames

```python
# Get the most recent frame (non-blocking)
frame_data = camera.get_latest_frame()

# Process a single frame directly (blocking)
frame_data = camera.get_one_frame_data()
```

## API Reference

### Gemini2 Class

- `__init__(buffer_size=5)`: Initialize with specified frame buffer size
- `init_camera()`: Configure and start the camera with default settings
- `start_pipeline()`: Start the camera pipeline
- `stop_pipeline()`: Stop the camera pipeline
- `start_acquisition_thread()`: Start background frame acquisition
- `stop_acquisition_thread()`: Stop background frame acquisition
- `get_latest_frame()`: Get the most recent frame (non-blocking)
- `get_one_frame_data()`: Get a single frame (blocking)
- `visualise_frame(color_image, depth_data)`: Display color and depth frames

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
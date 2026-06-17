# ESP32 Cam — YOLO Device Tracker

> **Work in progress**

Monitors a live ESP32 Cam MJPEG stream, detects laptops and mobile phones using YOLOv8, and tracks how long each device is visible per day. Daily totals are saved to a CSV file at midnight.

## How it works

- Polls the stream URL on startup and automatically reconnects if the stream goes down
- Runs YOLOv8n object detection on every 3rd frame (filtered to laptops + phones only)
- A bilateral filter is applied before inference to reduce JPEG compression noise from the ESP32 sensor
- Detection requires a rolling vote of 3 out of 5 consecutive YOLO frames to start a session (prevents false positives from artifacts)
- A 3-second grace period prevents brief detection gaps from splitting a continuous session — the gap is not counted in the total
- Time is accumulated in memory across stream reconnects; a CSV is written only when the date rolls over to the next day

## Setup

```bash
pip install -r requirements.txt
```

YOLOv8n weights (`yolov8n.pt`) are downloaded automatically on first run (~6 MB).

## Usage

```bash
python viewer.py
```

Press `q` to quit. The detection window shows bounding boxes for detected devices and a live session timer in the top-left corner.

## Output

One CSV file per day in the same directory:

```
detections_YYYY-MM-DD.csv
```

| Column | Description |
|---|---|
| `date` | Date of the recording |
| `device_type` | `laptop` or `cell_phone` |
| `total_seconds` | Cumulative on-screen time for the day |

## Hardware

- ESP32 Cam streaming MJPEG at `http://192.168.0.77:81/stream`
- Mac with Apple Silicon (MPS used for YOLO inference)

import csv
import time
from collections import defaultdict, deque
from datetime import datetime, date
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# --- Configuration -----------------------------------------------------------

STREAM_URL = "http://192.168.0.77:81/stream"

# COCO class IDs for the objects we care about
TRACKED = {63: "laptop", 67: "cell_phone"}

# How long (seconds) an object must be absent before its session is closed.
# The gap itself is not counted toward the daily total.
GRACE_SECONDS = 3

# Run YOLO inference every N frames; display still updates every frame.
DETECT_EVERY = 3

# Lower than the default 0.5 because JPEG compression reduces model confidence
# on genuine detections from the ESP32 sensor.
CONF_THRESHOLD = 0.35

# Bilateral filter parameters applied before inference to smooth JPEG block
# noise while preserving object edges (unlike a plain Gaussian blur).
BILATERAL_D = 5
BILATERAL_SIGMA = 60

# An object is only considered "present" when it appears in at least
# VOTE_THRESHOLD out of the last VOTE_WINDOW YOLO frames. This prevents
# single-frame compression artifacts from triggering false sessions.
VOTE_WINDOW = 5
VOTE_THRESHOLD = 3

OUTPUT_DIR = Path(__file__).parent

# --- Model setup -------------------------------------------------------------

# Use Apple Silicon GPU when available, otherwise fall back to CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)


# --- CSV helpers -------------------------------------------------------------

def save_daily_csv(day, totals):
    """Write aggregated on-screen totals for `day` to a CSV file."""
    path = OUTPUT_DIR / f"detections_{day}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "device_type", "total_seconds"])
        for device_name, secs in totals.items():
            if secs > 0:
                w.writerow([day, device_name, round(secs, 1)])
    print(f"Saved daily summary: {path}")


def check_day_rollover(daily_totals, current_date):
    """Save yesterday's totals to CSV if the calendar date has changed.

    Returns the updated current date (today).
    """
    today = date.today()
    if today != current_date:
        if any(v > 0 for v in daily_totals.values()):
            save_daily_csv(current_date, daily_totals)
        daily_totals.clear()
        return today
    return current_date


# --- Tracking helpers --------------------------------------------------------

def flush_tracking(tracking, daily_totals):
    """Close any open sessions and add their duration to the daily totals.

    Called when the stream dies or the user quits, so in-progress time is
    not lost before the next day-rollover save.
    """
    for name, s in tracking.items():
        if s["visible"] and s["last_seen"] is not None:
            daily_totals[name] += (s["last_seen"] - s["session_start"]).total_seconds()
            s["visible"] = False


# --- Stream helpers ----------------------------------------------------------

def wait_for_stream():
    """Block until the MJPEG stream returns a valid frame, then return the
    VideoCapture handle. Retries every 5 seconds so the script auto-starts
    whenever the ESP32 boots up.
    """
    print(f"Waiting for stream at {STREAM_URL} ...")
    while True:
        cap = cv2.VideoCapture(STREAM_URL)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("Stream live — starting detection.")
                return cap
        cap.release()
        time.sleep(5)


# --- Main detection loop -----------------------------------------------------

def run_detection(cap, daily_totals, current_date):
    """Read frames from `cap`, run YOLO, track device presence, and update
    `daily_totals`. Returns (quit_requested, current_date).

    Tracking state per device:
      visible      – whether the object is currently in a session
      session_start – when the current session began
      last_seen    – timestamp of the most recent confirmed detection
                     (used to measure the grace-period gap)
    """
    tracking = {
        n: {"visible": False, "session_start": None, "last_seen": None}
        for n in TRACKED.values()
    }
    frame_idx = 0
    quit_requested = False
    last_boxes = []  # bounding boxes from the most recent YOLO run, reused across frames
    votes = {n: deque(maxlen=VOTE_WINDOW) for n in TRACKED.values()}

    while True:
        ret, frame = cap.read()
        if not ret:
            # Stream dropped — exit loop so the caller can reconnect.
            break

        now = datetime.now()
        frame_idx += 1

        # --- YOLO inference (every DETECT_EVERY frames) ---
        if frame_idx % DETECT_EVERY == 0:
            # Denoise before inference; the filtered copy is never displayed.
            inference_frame = cv2.bilateralFilter(frame, BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA)
            results = model(inference_frame, verbose=False, conf=CONF_THRESHOLD,
                            classes=list(TRACKED.keys()))

            detected = set()
            last_boxes = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in TRACKED:
                        name = TRACKED[cls]
                        detected.add(name)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        last_boxes.append((x1, y1, x2, y2, name))

            # Update the vote window and confirm or start sessions.
            for name in TRACKED.values():
                votes[name].append(name in detected)
                confirmed = votes[name].count(True) >= VOTE_THRESHOLD
                s = tracking[name]
                if confirmed:
                    s["last_seen"] = now
                    if not s["visible"]:
                        # Object newly confirmed — begin a session.
                        s["visible"] = True
                        s["session_start"] = now

        # --- Day rollover check (runs every frame around midnight) ---
        current_date = check_day_rollover(daily_totals, current_date)

        # --- Grace-period check ---
        # If a tracked object has not been confirmed for GRACE_SECONDS, close
        # its session. Duration is measured to last_seen, not now, so the
        # gap is excluded from the total.
        for name, s in tracking.items():
            if s["visible"] and (now - s["last_seen"]).total_seconds() > GRACE_SECONDS:
                duration = (s["last_seen"] - s["session_start"]).total_seconds()
                daily_totals[name] += duration
                print(f"Session ended: {name} ({round(duration, 1)}s) | today total: {round(daily_totals[name], 1)}s")
                s["visible"] = False

        # --- Overlay: bounding boxes (reused from last YOLO run) ---
        for x1, y1, x2, y2, name in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Overlay: live session timer per device (top-left HUD) ---
        y = 25
        for name, s in tracking.items():
            if s["visible"]:
                secs = (now - s["session_start"]).total_seconds()
                cv2.putText(frame, f"{name}: {secs:.0f}s", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                y += 28

        cv2.imshow("ESP32 Cam - YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            quit_requested = True
            break

    # Fold any still-open sessions into the daily totals before returning.
    flush_tracking(tracking, daily_totals)
    return quit_requested, current_date


# --- Entry point -------------------------------------------------------------

def main():
    # daily_totals persists across stream reconnects so time is never lost.
    daily_totals = defaultdict(float)
    current_date = date.today()

    while True:
        current_date = check_day_rollover(daily_totals, current_date)
        cap = wait_for_stream()
        try:
            quit_requested, current_date = run_detection(cap, daily_totals, current_date)
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if quit_requested:
            print("Quit requested — exiting.")
            break

        print("Stream ended — waiting to reconnect...")
        time.sleep(5)


if __name__ == "__main__":
    main()

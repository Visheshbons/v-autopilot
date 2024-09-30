import cv2
import time
import argparse
import os
import glob
import numpy as np
import threading
import queue
from ultralytics import YOLO

# Required for BeamNG integration
from PIL import Image
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera

# --- Configuration for Control Display (Networking Removed) ---
STEERING_SCALE = 1.5  # Sensitivity factor for turning calculation (for display only)
DEFAULT_THROTTLE = 0.3 # Placeholder for throttle value (for display only)
CRITICAL_CLASS_ID = 0 # Assuming class 0 (e.g., 'stop sign') requires stopping calculation

# --- BeamNG/Queue Configuration ---
FRAME_QUEUE_MAXSIZE = 5
CV2_WINDOW_NAME = 'YOLO Autopilot Viewer'

# --- Helper Functions (Model Management) ---

def find_data_yaml(start_dir):
    matches = []
    for root, dirs, files in os.walk(start_dir):
        if 'data.yaml' in files:
            matches.append(os.path.join(root, 'data.yaml'))
    return matches


def find_latest_weights(runs_dir):
    patterns = [os.path.join(runs_dir, '**', 'best.pt'), os.path.join(runs_dir, '**', 'last.pt')]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    if not candidates:
        return None
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


# --- Control Logic Implementation (For Display Only) ---

def calculate_controls(results, frame_center_x):
    steering = 0.0
    throttle = DEFAULT_THROTTLE
    object_center_x = frame_center_x
    object_class_id = -1

    if not results or not results[0].boxes:
        return steering, throttle

    if results[0].masks is not None and len(results[0].masks) > 0:
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        mask_areas = np.sum(masks, axis=(1, 2))
        largest_idx = np.argmax(mask_areas)
        largest_mask = masks[largest_idx]
        object_class_id = classes[largest_idx]
        M = cv2.moments(largest_mask)
        if M["m00"] > 0:
            object_center_x = int(M["m10"] / M["m00"])
        else:
            boxes_xywh = results[0].boxes.xywh.cpu().numpy()
            object_center_x = boxes_xywh[largest_idx, 0]
    elif len(results[0].boxes) > 0:
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        areas = boxes_xywh[:, 2] * boxes_xywh[:, 3]
        largest_idx = np.argmax(areas)
        object_center_x = boxes_xywh[largest_idx, 0]
        object_class_id = classes[largest_idx]

    offset = object_center_x - frame_center_x
    normalized_steering = (offset / frame_center_x) * STEERING_SCALE
    steering = max(-1.0, min(1.0, normalized_steering))

    if object_class_id == CRITICAL_CLASS_ID:
        print(f"!!! CRITICAL OBJECT DETECTED (Class {object_class_id}) - Simulated Stop !!!")
        throttle = 0.0

    return steering, throttle


def draw_shaded_detections(frame_bgr, results, alpha=0.4):
    output_frame = frame_bgr.copy()
    overlay = output_frame.copy()

    if not results or not results[0].boxes:
        return output_frame

    class_names = results[0].names
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    color_map = {
        0: (0, 0, 255),
        1: (255, 255, 0),
        2: (0, 255, 0),
        3: (255, 0, 0),
        4: (0, 255, 255),
    }

    if results[0].masks is not None:
        polygons = results[0].masks.xy
        for i, poly in enumerate(polygons):
            cls_id = classes[i]
            color_bgr = color_map.get(cls_id, (255, 0, 255))
            cv2.fillPoly(overlay, [poly.astype(np.int32)], color_bgr)
    else:
        print("Warning: Model does not provide segmentation masks. Falling back to bounding boxes.")
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            cls_id = classes[i]
            color_bgr = color_map.get(cls_id, (255, 0, 255))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)

    cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box
        cls_id = classes[i]
        label = class_names.get(cls_id, f"Class {cls_id}")
        color_bgr = color_map.get(cls_id, (255, 0, 255))
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_frame, (x1, y1 - h - baseline - 5), (x1 + w + 10, y1), color_bgr, -1)
        cv2.putText(output_frame, label, (x1 + 5, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return output_frame


# --- BeamNG Capture Thread ---
def run_beamng_dashcam(frame_queue, fps=10, home=None, user=None):
    bng = BeamNGpy('localhost', 64256, home=home, user=user)
    try:
        bng.open()
    except Exception as e:
        print("Failed to connect to BeamNG:", e)
        frame_queue.put(None)
        return

    scenario = Scenario('west_coast_usa', 'autopilot_capture')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')
    scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()

    try:
        bng.traffic.spawn(max_amount=5)
    except Exception as e:
        print("Traffic spawn failed:", e)

    resolution_wide = (640, 240)
    camera_sensors = {}
    camera_sensors['front'] = Camera(
        'front_cam', bng, vehicle,
        is_render_colours=True,
        pos=(0, -2.0, 1.2),
        dir=(0, -1, 0),
        field_of_view_y=70,
        resolution=resolution_wide
    )
    vehicle.ai.set_mode('disabled')

    stop_event = threading.Event()
    def wait_for_enter():
        input('Press Enter to stop BeamNG capture...')
        stop_event.set()
    threading.Thread(target=wait_for_enter, daemon=True).start()

    interval = 1.0 / max(1, fps)
    time.sleep(1)
    try:
        while not stop_event.is_set():
            all_frames_numpy = {}
            for name, sensor in camera_sensors.items():
                dashcam_data = sensor.poll()
                if dashcam_data and 'colour' in dashcam_data and dashcam_data['colour'] is not None:
                    img_rgb = dashcam_data['colour'].convert('RGB')
                    all_frames_numpy[name] = np.array(img_rgb)
            if all_frames_numpy:
                try:
                    frame_queue.put(all_frames_numpy, block=False, timeout=0.01)
                except queue.Full:
                    pass
            time.sleep(interval)
    finally:
        bng.disconnect()
        frame_queue.put(None)


# --- Main Inference and Display ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default=None)
    parser.add_argument('--pretrained', '-p', default='yolov8n-seg.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--force-train', action='store_true')
    parser.add_argument('--input-mode', '-m', choices=['camera', 'beamng'], default='camera')
    parser.add_argument('--webcam-id', type=int, default=0)
    parser.add_argument('--home', default=None)
    parser.add_argument('--user', default=None)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(cwd, 'runs')
    best_weights = find_latest_weights(runs_dir) or args.pretrained
    detection_model = YOLO(best_weights)

    t = None
    cap = None
    frame_queue = None
    frame_width = 0
    frame_center_x = 0
    autopilot_enabled = False

    if args.input_mode == 'beamng':
        frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        t = threading.Thread(target=lambda: run_beamng_dashcam(frame_queue, fps=args.fps, home=args.home, user=args.user), daemon=True)
        t.start()
        first_frame_bundle = frame_queue.get()
        if first_frame_bundle is None:
            return
        frame_width = 640
        frame_center_x = frame_width // 2
    else:
        cap = cv2.VideoCapture(args.webcam_id)
        if not cap.isOpened():
            return
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_center_x = frame_width // 2

    try:
        while True:
            front_frame_bgr = None
            if args.input_mode == 'camera':
                ret, frame = cap.read()
                if not ret: break
                front_frame_bgr = cv2.flip(frame, 1)
            else:
                if 'first_frame_bundle' in locals():
                    frame_bundle = first_frame_bundle
                    del first_frame_bundle
                else:
                    try:
                        frame_bundle = frame_queue.get(timeout=0.001)
                    except queue.Empty:
                        if t and not t.is_alive(): break
                        continue
                if frame_bundle is None: break
                if 'front' in frame_bundle:
                    front_frame_rgb = frame_bundle['front']
                    front_frame_bgr = cv2.cvtColor(front_frame_rgb, cv2.COLOR_RGB2BGR)

            if front_frame_bgr is not None:
                results = detection_model.predict(source=front_frame_bgr, imgsz=args.imgsz, verbose=False)

                raw_frame = front_frame_bgr.copy()
                vision_frame = draw_shaded_detections(front_frame_bgr, results)
                overlay_only = np.zeros_like(front_frame_bgr)
                overlay_only = draw_shaded_detections(overlay_only, results)

                if autopilot_enabled:
                    steering, throttle = calculate_controls(results, frame_center_x)
                    autopilot_status = "ON (Press 'a' to stop)"
                    status_color = (0, 255, 0)
                else:
                    steering, throttle = 0.0, 0.0
                    autopilot_status = "OFF (Press 'a' to start)"
                    status_color = (0, 0, 255)

                cv2.putText(vision_frame, f"Autopilot: {autopilot_status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(vision_frame, f"Steering: {steering:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vision_frame, f"Throttle: {throttle:.3f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.line(vision_frame, (frame_center_x, 0), (frame_center_x, vision_frame.shape[0]), (255, 0, 0), 1)

                stacked_display = np.vstack([raw_frame, vision_frame, overlay_only])
                cv2.imshow(CV2_WINDOW_NAME, stacked_display)
            else:
                time.sleep(0.01)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('a'):
                autopilot_enabled = not autopilot_enabled
                print("Autopilot", "ON" if autopilot_enabled else "OFF")
    finally:
        if cap: cap.release()
        if t and t.is_alive(): t.join(timeout=2)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

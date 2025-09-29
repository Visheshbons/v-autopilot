from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Clean exit on ^C
try:
    results = model(source=0, show=True)
except KeyboardInterrupt:
    print("Exiting...")
    exit(0)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
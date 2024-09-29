from ultralytics import YOLO
import argparse
import threading
import time
from pathlib import Path
import queue
import numpy as np 
import cv2 

# Pillow is required to convert image data
from PIL import Image
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera 

# Define the maximum size for our in-memory queue
FRAME_QUEUE_MAXSIZE = 5
CV2_WINDOW_NAME = 'Autonomous Perception System (4-Camera View)'

# No border/label function, as requested.

def run_beamng_dashcam(frame_queue, fps=10, duration=None, home=None, user=None):
    """
    Starts BeamNG, runs a scenario, and captures frames from four dashcam sensors, 
    placing them into an in-memory queue as NumPy arrays within a dictionary.
    """
    
    # Launch BeamNGpy
    bng = BeamNGpy('localhost', 25252, home=home, user=user)
    try:
        bng.open()
    except Exception as e:
        # **PROMINENT ERROR MESSAGE FOR CONNECTION FAILURE**
        print("\n\n#####################################################")
        print("!! CRITICAL ERROR: FAILED TO CONNECT TO BEAMNG !!")
        print(f"Error: {e}")
        print("Please ensure the BeamNG.drive simulation is running **before** executing this script.")
        print("Also double-check the 'home' and 'user' paths in your command line arguments.")
        print("#####################################################\n")
        frame_queue.put(None) # Signal main thread to stop
        return

    # Setup scenario
    scenario = Scenario('west_coast_usa', 'autopilot_capture')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')
    scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795)) 
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()

    # --- FIX for Lua BNGError: Traffic Initialization ---
    try:
        print("Attempting to initialize and spawn traffic...")
        # Max vehicles set to 5, feel free to adjust this number
        bng.traffic.spawn(max_amount=5)
        print("Traffic successfully spawned.")
    except Exception as traffic_e:
        print(f"WARNING: Failed to setup or spawn traffic. Error: {traffic_e}")
        print("The vision system will run, but you may have no other vehicles on the road.")
    # -----------------------------------------------------

    # --- Sensor Setup ---
    resolution_side = (320, 240) # Standard resolution for side cameras
    resolution_wide = (640, 240) # Wide resolution for front/rear (to prevent warping)

    # NOTE: Using the user-provided, corrected camera positions.
    sensor_specs = {
        # Front and Rear now use the wider 640x240 resolution
        'front': {'pos': (0, -2.0, 1.2), 'dir': (0, -1, 0), 'fov': 70, 'res': resolution_wide},     
        'rear': {'pos': (0, 3.5, 1.2), 'dir': (0, 1, 0), 'fov': 70, 'res': resolution_wide},      
        
        # Left and Right remain at 320x240
        'left': {'pos': (-1.0, 0.5, 1.2), 'dir': (-1, 0, 0), 'fov': 90, 'res': resolution_side},    
        'right': {'pos': (1.0, 0.5, 1.2), 'dir': (1, 0, 0), 'fov': 90, 'res': resolution_side},     
    }

    camera_sensors = {}

    # 1. Create and attach all four Camera sensors
    for name, spec in sensor_specs.items():
        try:
            camera_sensors[name] = Camera(
                f'{name}_cam', bng, vehicle, 
                is_render_colours=True,
                pos=spec['pos'],  
                dir=spec['dir'],        
                field_of_view_y=spec['fov'], 
                resolution=spec['res'] # Use specific resolution
            )
        except Exception as e:
            print(f"Error creating {name} camera sensor: {e}")
            frame_queue.put(None)
            return

    # Let player control vehicle
    vehicle.ai.set_mode('disabled')

    # Capture loop setup
    stop_event = threading.Event()
    def wait_for_enter():
        input('Press Enter to stop BeamNG capture...')
        stop_event.set()

    # Start the thread to wait for user input
    threading.Thread(target=wait_for_enter, daemon=True).start()

    interval = 1.0 / max(1, fps)
    
    # Give BeamNG a moment to load everything before starting capture
    time.sleep(1) 
    
    try:
        while not stop_event.is_set():
            
            all_frames_numpy = {}
            
            # 2. Poll the sensor data for all four cameras
            for name, sensor in camera_sensors.items():
                try:
                    dashcam_data = sensor.poll()
                except AttributeError:
                    print(f"Critical Error: {name} sensor object has no 'poll' method. Cannot fetch frame data.")
                    stop_event.set()
                    break

                # 3. Extract the image object, convert to NumPy
                if 'colour' in dashcam_data:
                    image_data = dashcam_data['colour']
                    
                    try:
                        # Convert Pillow image (RGB mode) to NumPy array
                        img_rgb = image_data.convert('RGB')
                        image_array = np.array(img_rgb)
                        all_frames_numpy[name] = image_array
                        
                    except AttributeError:
                        print("Critical Error: Image data returned is not in an expected format (Pillow Image).")
                        stop_event.set()
                        break
            
            if not all_frames_numpy: # Check if loop broke or no frames collected
                time.sleep(interval)
                continue

            try:
                # Put the dictionary of NumPy arrays into the queue
                frame_queue.put(all_frames_numpy, block=False, timeout=0.01)
            except queue.Full:
                # If the queue is full, the processing thread is too slow. Skip this frame.
                pass 
                
            time.sleep(interval)
            
    finally:
        # Cleanup and disconnect
        print("Stopping BeamNG capture and disconnecting.")
        try:
            bng.disconnect()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        # Signal the main thread that the producer is done
        frame_queue.put(None) 


def main():
    parser = argparse.ArgumentParser(description='Run YOLO on BeamNG dashcam or webcam.')
    parser.add_argument('--source', '-s', default=0, help='Webcam source (0 for default camera)')
    parser.add_argument('--input-mode', '-m', choices=['camera', 'beamng'], default='camera')
    parser.add_argument('--home', default=r'C:\Users\Vishesh\Desktop\BeamNG.tech.v0.35.5.0', help='BeamNG home folder')
    parser.add_argument('--user', default=r'C:\Users\Vishesh\Desktop\Code\Autopilot\userFolder', help='BeamNG userFolder')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for dashcam')
    args = parser.parse_args()

    # Initialize YOLO model
    model = YOLO('yolov8n.pt')
    t = None # Initialize thread variable
    
    if args.input_mode == 'beamng':
        frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)

        t = threading.Thread(
            target=lambda: run_beamng_dashcam(
                frame_queue,
                fps=args.fps,
                home=args.home,
                user=args.user
            ),
            daemon=True
        )
        t.start()

        print("BeamNG multi-camera feed is starting. Running YOLO on 4 in-memory streams...")
        print("Waiting for the first frame bundle...")
        
        # --- UI Geometry Constants ---
        # Front/Rear are captured at 640x240, Side at 320x240.
        ROW_H = 240 # Height of all rows
        TOTAL_W = 640 # Total window width
        SIDE_W_DISPLAY = 310 # Width of side views (slightly resized from 320)
        GAP_W = TOTAL_W - (SIDE_W_DISPLAY * 2) # 640 - 620 = 20 (Center gap)
        # --- End UI Geometry Constants ---

        try:
            # Block indefinitely for the very first frame/signal
            first_frame_bundle = frame_queue.get() 
            
            if first_frame_bundle is None:
                print("Connection to BeamNG failed, shutting down.")
                return

            print("Starting YOLO inference...")
            
            # Use a continuous loop for YOLO processing and OpenCV display
            while True:
                # Use the first bundle if available, otherwise get from queue
                frame_bundle = first_frame_bundle if 'first_frame_bundle' in locals() else None 

                if frame_bundle is None:
                    try:
                        frame_bundle = frame_queue.get(timeout=0.001) 
                    except queue.Empty:
                        if not t.is_alive():
                            print("BeamNG capture thread disconnected, stream ended.")
                            break
                        continue 
                
                # Clear the first_frame_bundle variable after its first use
                if 'first_frame_bundle' in locals():
                    del first_frame_bundle
                
                if frame_bundle is None:
                    break
                
                # --- YOLO Processing ---
                # This dictionary stores the annotated NumPy arrays (in RGB format)
                annotated_frames = {}
                for name, frame_array in frame_bundle.items():
                    # 1. Run YOLO inference
                    results = model.predict(source=frame_array, verbose=False)
                    
                    # 2. Get the annotated image (NumPy array) - YOLO outputs RGB array
                    annotated_frames[name] = results[0].plot()

                # --- UI Creation (Custom 3-Row Layout) ---
                
                # Convert all annotated frames (RGB from YOLO) to BGR (for OpenCV concatenation/display)
                annotated_frames_bgr = {
                    name: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                    for name, frame in annotated_frames.items()
                }

                # ---------------------
                # 5. Row 1: Full-width Front Camera (640 x 240)
                # ---------------------
                # The front camera is already 640x240, no resize needed.
                row1 = annotated_frames_bgr['front']
                
                # ---------------------
                # 6. Row 2: Left | Center Gap | Right (Total 640 x 240)
                # ---------------------
                
                # Resize side cameras to fit the display width (310x240)
                row2_left = cv2.resize(annotated_frames_bgr['left'], (SIDE_W_DISPLAY, ROW_H), interpolation=cv2.INTER_LINEAR)
                row2_right = cv2.resize(annotated_frames_bgr['right'], (SIDE_W_DISPLAY, ROW_H), interpolation=cv2.INTER_LINEAR)
                
                # Create the center gap (Black color)
                center_pad = np.full((ROW_H, GAP_W, 3), 0, dtype=np.uint8) 
                
                # Concatenate the middle row
                row2 = cv2.hconcat([row2_left, center_pad, row2_right])

                # ---------------------
                # 7. Row 3: Full-width Rear Camera (640 x 240)
                # ---------------------
                # The rear camera is already 640x240, no resize needed.
                row3 = annotated_frames_bgr['rear']
                
                # ---------------------
                # 8. Final Assembly: Concatenate all three rows vertically
                # ---------------------
                composite_frame = cv2.vconcat([row1, row2, row3])
                
                # 9. Display the annotated frame using OpenCV
                cv2.imshow(CV2_WINDOW_NAME, composite_frame)
                
                # 10. Check for exit key (q)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Exiting YOLO inference...")
        except Exception as e:
            print(f"An error occurred during YOLO inference: {e}")
            
        finally:
            # Clean up the OpenCV window
            cv2.destroyAllWindows() 

            # Safely ensure the capture thread is stopped
            if t and t.is_alive():
                 print("Waiting for BeamNG capture thread to gracefully finish...")
                 t.join(timeout=2) 
            
    elif args.input_mode == 'camera':
        # Simple camera/directory mode (still uses stream=True from YOLO)
        source = int(args.source) if str(args.source).isdigit() else args.source
        
        if source is not None:
             try:
                print("Starting YOLO inference on external source...")
                results = model(source=source, show=True, stream=True)
                for r in results:
                    pass 
             except Exception as e:
                print(f"An error occurred during YOLO inference: {e}")

if __name__ == '__main__':
    main()
from ultralytics import YOLO
import argparse
import threading
import time
from pathlib import Path
import queue
import numpy as np 
import cv2 # <-- NEW: For continuous real-time display without flicker

# Pillow is required to convert image data
from PIL import Image
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera 

# Define the maximum size for our in-memory queue
FRAME_QUEUE_MAXSIZE = 5
CV2_WINDOW_NAME = 'YOLOv8 Real-Time Detection (BeamNG)'

def run_beamng_dashcam(frame_queue, fps=10, duration=None, home=None, user=None):
    """
    Start BeamNG, run a scenario, and capture dashcam frames, placing them 
    directly into an in-memory queue as NumPy arrays.
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
    # Use a safe starting position and ensure rotation is (0, 0, 0, 1) or close to default
    scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795)) 
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()

    sensor_id = 'dashcam'
    
    # 1. Create the Camera sensor
    try:
        # Fix: Adjusted position slightly back and centered the direction.
        # dir=(0, 1, 0) is typically forward along the vehicle's local Y-axis.
        camera_sensor = Camera(
            sensor_id, bng, vehicle, 
            is_render_colours=True,
            pos=(-0.2, -1.5, 1.2),  # Position: slightly left, FORWARD (1.5m), and up (1.2m)
            dir=(0, -1, 0),        # Direction: looking straight forward
            field_of_view_y=70, 
            resolution=(640, 480) 
        )
    except Exception as e:
        print(f"Error creating Camera sensor: {e}")
        frame_queue.put(None) # Signal main thread to stop
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
            
            # 2. Poll the sensor data
            try:
                dashcam_data = camera_sensor.poll()
            except AttributeError:
                print("Critical Error: camera_sensor object has no 'poll' method. Cannot fetch frame data.")
                stop_event.set()
                break

            # 3. Extract the image object, convert to NumPy, and put in queue
            if 'colour' in dashcam_data:
                image_data = dashcam_data['colour']
                
                try:
                    # Convert Pillow image (RGB mode) to NumPy array for YOLO
                    img_rgb = image_data.convert('RGB')
                    image_array = np.array(img_rgb)
                    
                    # Put the NumPy array into the queue
                    # Use block=False and timeout to handle a full queue gracefully
                    frame_queue.put(image_array, block=False, timeout=0.01)
                    
                except AttributeError:
                    print("Critical Error: Image data returned is not in an expected format (Pillow Image).")
                    stop_event.set()
                    break
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
        # Create the thread-safe queue for frames
        frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)

        # Start BeamNG dashcam in background
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

        print("BeamNG dashcam feed is starting. Running YOLO on in-memory frames...")
        print("Waiting for the first frame...")
        
        try:
            # Block indefinitely for the very first frame/signal to ensure the connection attempt completes
            first_frame = frame_queue.get() 
            
            # Check for the sentinel value (None) immediately after the first block
            if first_frame is None:
                print("Connection to BeamNG failed, shutting down.")
                return

            print("Starting YOLO inference...")
            
            # Use a continuous loop for YOLO processing and OpenCV display
            while True:
                # Put the first frame back into the processing stream
                frame_array = first_frame if 'first_frame' in locals() else None 

                # Process subsequent frames from the queue
                if frame_array is None:
                    try:
                        frame_array = frame_queue.get(timeout=0.001) # Very short timeout for responsiveness
                    except queue.Empty:
                        if not t.is_alive():
                            print("BeamNG capture thread disconnected, stream ended.")
                            break
                        continue # Keep waiting if thread is alive and queue is empty
                
                # If we processed the first frame, clear the variable so we proceed to the queue next time
                if 'first_frame' in locals():
                    del first_frame
                
                # Check for the sentinel value (None) indicating the producer thread is done
                if frame_array is None:
                    break
                
                # 1. Run YOLO inference (no show=True here, we display manually)
                # We use stream=True, even for single images, to potentially optimize the backend
                results = model.predict(source=frame_array, verbose=False)

                # 2. Get the annotated image (NumPy array) from the results
                # YOLO returns a list of results (one per batch item, which is one image here)
                annotated_frame = results[0].plot()

                # 3. Display the annotated frame using OpenCV
                cv2.imshow(CV2_WINDOW_NAME, annotated_frame)
                
                # 4. Check for exit key (q)
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
                # Note: YOLO's native stream=True handles display better for files/webcams
                results = model(source=source, show=True, stream=True)
                for r in results:
                    pass 
             except Exception as e:
                print(f"An error occurred during YOLO inference: {e}")

if __name__ == '__main__':
    main()
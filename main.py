# main.py

from beamngpy import BeamNGpy, Scenario, Vehicle
import os
import time
import threading
from pathlib import Path


def run_beamng_dashcam(output_dir, fps=10, duration=None, host='localhost', port=25252, home=None, user=None, beamng_exe=None):
    """Start BeamNG, load a demo scenario with a vehicle and capture dashcam screenshots.

    Args:
        output_dir (str): Folder where frames will be saved (created if missing).
        fps (int): Frames per second to capture.
        duration (float|None): Seconds to run. If None, runs until user presses Enter.
        host (str): BeamNG host.
        port (int): BeamNG port.
        home (str|None): BeamNG home path (optional, needed if BeamNG should be started automatically).
        user (str|None): BeamNG userFolder path (optional).
        beamng_exe (str|None): Explicit BeamNG exe (not strictly needed; BeamNGpy can use `home`).
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create BeamNGpy instance
    bng = BeamNGpy(host, port, home=home, user=user)
    bng.open()

    # Setup scenario
    scenario = Scenario('west_coast_usa', 'autopilot_capture')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')

    scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.make(bng)

    bng.scenario.load(scenario)
    bng.scenario.start()

    vehicle.ai.set_mode('traffic')

    stop_event = threading.Event()

    def wait_for_enter():
        try:
            input('Press Enter to stop BeamNG capture...')
        except Exception:
            pass
        stop_event.set()

    if duration is None:
        t = threading.Thread(target=wait_for_enter, daemon=True)
        t.start()
    else:
        def timer():
            time.sleep(duration)
            stop_event.set()
        t = threading.Thread(target=timer, daemon=True)
        t.start()

    # Capture loop
    interval = 1.0 / max(1, fps)
    frame = 0
    try:
        while not stop_event.is_set():
            fname = output_path / f"frame_{frame:06d}.png"
            try:
                bng.screenshot(str(fname))
            except Exception:
                try:
                    vehicle.screenshot(str(fname))
                except Exception:
                    pass
            frame += 1
            time.sleep(interval)
    finally:
        try:
            bng.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    # Default paths if you want it to auto-launch BeamNG
    default_home = r'C:\Users\Vishesh\Desktop\BeamNG.tech.v0.35.5.0'
    default_user = r'C:\Users\Vishesh\Desktop\BeamNG.tech.v0.35.5.0\userFolder'
    print('Starting BeamNG demo. Frames will not be saved unless you call run_beamng_dashcam directly.')

    bng = BeamNGpy('localhost', 25252, home=default_home, user=default_user)
    bng.open()
    scenario = Scenario('west_coast_usa', 'example')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')
    scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    vehicle.ai.set_mode('disabled')

    run_beamng_dashcam(r'C:\Users\Vishesh\Desktop\Code\Autopilot\frames')

    input('Hit Enter when done...')
    bng.disconnect()
import os
import glob
import argparse
import time
import threading
import tempfile
from ultralytics import YOLO

try:
    # local import to avoid heavy dependency if not using beamng option
    from main import run_beamng_dashcam
except Exception:
    run_beamng_dashcam = None

# Train a YOLOv8 model on a dataset (detected via data.yaml) and then run inference
# Behavior:
# - Auto-detects a data.yaml in the workspace if --dataset is not provided
# - Skips training if existing trained weights (best.pt or last.pt) are found under ./runs/ unless --force-train is given
# - Trains using a pretrained backbone (default: yolov8n.pt)
# - After training (or if weights already exist) loads the best weights and runs inference


def find_data_yaml(start_dir):
    matches = []
    for root, dirs, files in os.walk(start_dir):
        if 'data.yaml' in files:
            matches.append(os.path.join(root, 'data.yaml'))
    return matches


def find_latest_weights(runs_dir):
    # look for best.pt first, then last.pt
    patterns = [os.path.join(runs_dir, '**', 'best.pt'), os.path.join(runs_dir, '**', 'last.pt')]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    if not candidates:
        return None
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on a local dataset and run inference')
    parser.add_argument('--dataset', '-d', help='Path to dataset folder or data.yaml (optional)', default=None)
    parser.add_argument('--pretrained', '-p', default='yolov8n.pt', help='Pretrained model to use for transfer learning')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--force-train', action='store_true', help='Force training even if weights already exist')
    parser.add_argument('--source', '-s', default=0, help='Inference source (0 for webcam, path to image/video/folder)')
    parser.add_argument('--input-mode', '-m', choices=['camera', 'beamng'], default='camera', help="Choose input: 'camera' for webcam, 'beamng' to run BeamNG dashcam frames")
    parser.add_argument('--runasdate-path', default=None, help='Path to RunAsDate.exe to launch BeamNG with a fake date')
    parser.add_argument('--runasdate-date', default=None, help='Date to run BeamNG as (e.g. 16/05/2025)')
    parser.add_argument('--beamng-exe', default=None, help='Explicit path to BeamNG exe if auto-detection fails')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for inference')
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(cwd, 'runs')

    # Locate data.yaml
    data_yaml = None
    if args.dataset:
        # allow passing either the data.yaml file or the dataset directory containing it
        if os.path.isfile(args.dataset) and os.path.basename(args.dataset).lower() == 'data.yaml':
            data_yaml = os.path.abspath(args.dataset)
        elif os.path.isdir(args.dataset):
            candidate = os.path.join(args.dataset, 'data.yaml')
            if os.path.isfile(candidate):
                data_yaml = os.path.abspath(candidate)
            else:
                raise FileNotFoundError(f"No data.yaml found in provided dataset directory {args.dataset}")
        else:
            raise FileNotFoundError(f"Provided dataset path {args.dataset} not found")
    else:
        matches = find_data_yaml(cwd)
        if len(matches) == 0:
            raise FileNotFoundError('No data.yaml found in workspace. Please pass --dataset pointing to your dataset folder or data.yaml.')
        if len(matches) > 1:
            print(f"Multiple data.yaml files found, picking the first one: {matches[0]}")
        data_yaml = matches[0]

    print(f"Using data.yaml: {data_yaml}")

    # See if we already have trained weights in ./runs/
    existing_weights = find_latest_weights(runs_dir)
    if existing_weights and not args.force_train:
        print(f"Found existing trained weights: {existing_weights}. Skipping training.")
        best_weights = existing_weights
    else:
        # Train
        project = os.path.join(runs_dir, 'train')
        name = 'autopilot'
        print(f"Training new model using pretrained weights: {args.pretrained}")
        model = YOLO(args.pretrained)
        try:
            model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=project, name=name, exist_ok=True)
        except Exception as e:
            print(f"Training failed: {e}")
            raise

        # find generated weights
        best_weights = find_latest_weights(runs_dir)
        if not best_weights:
            raise FileNotFoundError('Training completed but no weights (best.pt or last.pt) were found under runs/')
        print(f"Training finished. Best weights found at: {best_weights}")

    # Load best weights and run inference
    print(f"Loading model weights from {best_weights} for inference.")
    detection_model = YOLO(best_weights)
    try:
        # Determine source based on input-mode
        source = args.source
        if args.input_mode == 'beamng':
            if run_beamng_dashcam is None:
                raise RuntimeError('BeamNG support is not available (failed to import run_beamng_dashcam from main).')

            # create a temp directory for beamng frames
            tmpdir = tempfile.mkdtemp(prefix='beamng_frames_')

            # start BeamNG capture in a background thread
            def start_beamng():
                try:
                    # run for indefinite duration; user can stop by pressing Enter in BeamNG thread
                    run_beamng_dashcam(tmpdir, fps=10, duration=None, runasdate_path=args.runasdate_path, runasdate_date=args.runasdate_date, beamng_exe=args.beamng_exe)
                except Exception as e:
                    print(f'BeamNG capture failed: {e}')

            t = threading.Thread(target=start_beamng, daemon=True)
            t.start()

            # point YOLO source to the folder of frames (it will watch new images)
            source = tmpdir
            print(f'BeamNG dashcam frames will be written to: {tmpdir}. Running YOLO on this folder...')
        else:
            # For webcam use source=0 (int), argparse provides string so cast if appropriate
            if isinstance(source, str) and source.isdigit():
                source = int(source)
        detection_model(source=source, show=True, conf=args.conf)
    except KeyboardInterrupt:
        print('Exiting...')
        exit(0)
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        exit(1)


if __name__ == '__main__':
    main()
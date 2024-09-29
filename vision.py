from ultralytics import YOLO
import argparse
import os
import glob
import time

# --- Helper Functions ---

def find_data_yaml(start_dir):
    """
    Recursively searches for 'data.yaml' starting from the given directory.
    This helps locate the dataset configuration file easily.
    """
    matches = []
    # os.walk efficiently traverses the directory tree
    for root, dirs, files in os.walk(start_dir):
        if 'data.yaml' in files:
            # Found a match!
            matches.append(os.path.join(root, 'data.yaml'))
    return matches


def find_latest_weights(runs_dir):
    """
    Searches the YOLO 'runs' directory for the most recently modified 
    'best.pt' or 'last.pt' weight file.
    """
    # Look for best.pt first, then last.pt in all subdirectories of runs_dir
    patterns = [os.path.join(runs_dir, '**', 'best.pt'), os.path.join(runs_dir, '**', 'last.pt')]
    candidates = []
    
    for pat in patterns:
        # glob.glob is used with recursive=True to find files deep inside the 'runs' folder
        candidates.extend(glob.glob(pat, recursive=True))
        
    if not candidates:
        return None
        
    # Sort candidates by modification time (most recent first)
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Return the path to the newest weight file
    return candidates[0]


# --- Main Logic ---

def main():
    """
    Parses arguments, locates the dataset, determines if training is needed, 
    and then executes the YOLOv8 training command.
    """
    parser = argparse.ArgumentParser(description='Train YOLOv8 on a local dataset.')
    
    # Arguments for dataset location and training parameters
    parser.add_argument('--dataset', '-d', help='Path to dataset folder or data.yaml (optional)', default=None)
    parser.add_argument('--pretrained', '-p', default='yolov8n.pt', help='Pretrained model to use for transfer learning (e.g., yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size for training.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--force-train', action='store_true', help='Force training even if existing weights are found.')
    
    args = parser.parse_args()

    # Determine paths
    cwd = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(cwd, 'runs')

    # 1. Locate data.yaml (Dataset Configuration)
    data_yaml = None
    if args.dataset:
        # Check if user provided a path
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
        # Automatically search for data.yaml in the current directory (for Roboflow structures)
        matches = find_data_yaml(cwd)
        if len(matches) == 0:
            raise FileNotFoundError('No data.yaml found in workspace. Please ensure the script is in the dataset folder or pass --dataset.')
        if len(matches) > 1:
            print(f"Multiple data.yaml files found, picking the first one: {matches[0]}")
        data_yaml = matches[0]

    print(f"Using data.yaml: {data_yaml}")
    # 

    # 2. Check for Existing Weights
    existing_weights = find_latest_weights(runs_dir)
    
    if existing_weights and not args.force_train:
        print(f"Found existing trained weights: {existing_weights}. Skipping training.")
        best_weights = existing_weights
    else:
        # 3. Execute Training
        project = os.path.join(runs_dir, 'train')
        name = 'autopilot'
        print(f"Starting new model training using pretrained weights: {args.pretrained}")
        
        # Load the base model (e.g., yolov8n.pt)
        model = YOLO(args.pretrained)
        
        try:
            # Run the training process
            model.train(
                data=data_yaml, 
                epochs=args.epochs, 
                imgsz=args.imgsz, 
                batch=args.batch, 
                project=project, 
                name=name, 
                exist_ok=True
            )
        except Exception as e:
            print(f"Training failed. Make sure you have the 'ultralytics' library installed and GPU drivers are up to date (if using a GPU). Error: {e}")
            raise

        # 4. Confirm Trained Weights Location
        best_weights = find_latest_weights(runs_dir)
        if not best_weights:
            raise FileNotFoundError('Training completed but no weights (best.pt or last.pt) were found under runs/. Check training logs.')
        print(f"Training finished successfully. Best weights found at: {best_weights}")

    # 5. Load the Trained Model (Ready for Inference)
    print(f"Loading final model weights from {best_weights} for potential future inference.")
    
    try:
        detection_model = YOLO(best_weights)
        print("Model loaded successfully. The script will now exit.")
        # If you wanted to run a quick test inference here, you would add it now.
        
    except Exception as e:
        print(f"An error occurred while loading the final model: {e}")
        exit(1)


if __name__ == '__main__':
    main()
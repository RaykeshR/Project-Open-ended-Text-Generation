import torch
import sys

try:
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", file=sys.stdout)               #True
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}", file=sys.stdout)           #1
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}", file=sys.stdout)       #0
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}", file=sys.stdout)   #NVIDIA GeForce ...
except Exception as e:
    print(f"An error occurred: {e}", file=sys.stderr)
    sys.exit(1)
sys.exit(0)

from ultralytics import YOLO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    model = YOLO("yolo11n.pt")
    results = model.train(data="data.yaml", epochs=1000)

if __name__ == "__main__":
    main()
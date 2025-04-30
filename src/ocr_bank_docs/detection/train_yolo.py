from ultralytics import YOLO

def train_yolo(data_yaml: str, weights_out: str, epochs=50, imgsz=640, batch=16):
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/detect",
        name="yolov8_handtext",
        pretrained=True,
    )
    model.model.save(weights_out)

if __name__ == "__main__":
    import argparse, pathlib, shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--out", required=True, help="where to save best.pt")
    args = parser.parse_args()
    train_yolo(args.data, args.out)

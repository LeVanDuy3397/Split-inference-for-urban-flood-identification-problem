import os
from ultralytics import YOLO


def main():
    # Cấu hình căn bản
    data_yaml = "data.yaml"  # cập nhật đường dẫn nếu để nơi khác
    # Chọn kiến trúc:
    # - Detect: "yolov8n.pt", "yolov8s.pt", ...
    # - Segment: "yolov8n-seg.pt", "yolov8s-seg.pt", ...
    model_name = "yolov8n.pt"  # đổi thành "yolov8n-seg.pt" nếu bạn muốn segmentation
    # Tạo/Load model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=100,            # chỉnh theo dataset
        imgsz=640,             # kích thước ảnh train
        batch=16,              # chỉnh theo VRAM
        device="cpu",          # GPU id; dùng "cpu" nếu không có GPU
        workers=4,             # số luồng dataloader
        optimizer="auto",      # để auto chọn; hoặc "SGD"/"AdamW"
        lr0=0.01,              # lr khởi đầu (tuỳ chỉnh)
        weight_decay=0.0005,
        patience=50,           # early stopping
        project="runs",        # thư mục gốc
        name="train",          # tên run
        exist_ok=True,         # ghi đè nếu tồn tại
        verbose=True
    )

    # Val sau train (Ultralytics sẽ tự val trong quá trình train; đây là val riêng nếu muốn)
    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device="cpu"
    )
    print("Validation metrics:", metrics)

    # Đường dẫn weights
    # best.pt được lưu tại: runs/detect/train/weights/best.pt (task detect)
    # hoặc: runs/segment/train/weights/best.pt (task segment)
    weights_dir = model.trainer.best if hasattr(model, "trainer") else None
    print("Best weights saved at:", weights_dir)


main()

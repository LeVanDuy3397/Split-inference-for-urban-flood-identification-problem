import os
from glob import glob
from ultralytics import YOLO


def main():
    # Trỏ tới best weights sau train
    # Với detect:
    weights = "yolov8n.pt"  # yolov8_version1.pt
    # Nếu dùng segmentation: "runs/segment/train/weights/best.pt"
    if not os.path.exists(weights):
        print(f"Không tìm thấy weights: {weights}")
    model = YOLO(weights)

    # Thư mục ảnh test (có thể dùng valid/images hoặc một folder test riêng)
    source_dir = "dataset/test/images"  # chỉnh theo nhu cầu
    image_paths = sorted(glob(os.path.join(source_dir, "*.*")))
    if len(image_paths) == 0:
        print(f"Không thấy ảnh trong: {source_dir}")

    # Inference hàng loạt
    results = model.predict(
        source=source_dir,
        imgsz=640,
        conf=0.2,        # ngưỡng confidence
        iou=0.45,         # IoU NMS
        device="cpu",         # "cpu" nếu không có GPU
        save=True,        # lưu ảnh vẽ kết quả
        save_txt=True,    # lưu nhãn YOLO predicted
        project="runs",
        name="predict",
        exist_ok=True,
        verbose=True
    )

    # In ra tóm tắt một vài kết quả
    for i, r in enumerate(results[:5]):
        # r.boxes.data: [x1,y1,x2,y2,score,cls]
        print(
            f"Image {i}: shape={r.orig_shape}, det={len(r.boxes)}, data={r.boxes.data.cpu().numpy()}")


main()

from email import message
from ultralytics.utils import ops
from ultralytics.engine.results import Results
import os
import cv2
import sys
import base64
import pika
import pickle
import torch
import torch.nn as nn
import numpy as np
from src.Model import SplitDetectionPredictor
import src.Log
import json
import base64


def display_top_5_scores(p0):
    if torch.is_tensor(p0) and p0.ndim == 3:
        # Tách các thông số từ tensor p0
        scores = p0[:, 4:8, :]  # Lấy 4 thông số score cuối cùng

        # Lặp qua từng class id (0 đến 3)
        for class_id in range(4):
            # Lấy 5 score cao nhất của class id hiện tại
            top_5_scores, _ = torch.topk(scores[:, class_id, :], 5, dim=-1)

            # In ra kết quả
            print(f"Top 5 scores for class {class_id}:")
            print(top_5_scores.squeeze())
            print()
    else:
        print("Input tensor is not valid.")


class Server:
    def __init__(self, config):
        # RabbitMQ
        # truy cập vào RabbitMQ và bên trong là address
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        # truy cập vào server và bên trong là model
        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_module = config["server"]["cut-module"]
        self.batch_frame = config["server"]["batch-frame"]

        # ds lưu mỗi phần tử là 1 mảng 2 chiều các dự đoán, mỗi dự đoán sẽ có 6 giá trị
        self.predictions = []
        self.Processed_predictions = []
        # ds lưu mỗi phần tử là 1 mảng 2 chiều các dự đoán và 1 mảng 2 chiều chứa vị trí kinh, vĩ, thời gian
        self.predictionAndlocation = []

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')
        self.channel.queue_declare(queue='timer_queue')
        self.channel.queue_declare(queue='prediction_queue', durable=True)
        self.channel.queue_declare(queue='location_queue', durable=True)
        self.channel.queue_declare(
            queue='prediction_and_location', durable=True)
        # chạy qua 3 client, rồi tạo ra 1 list [0,0]
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(
            queue='rpc_queue', on_message_callback=self.on_request)
        self.channel.basic_consume(
            queue='timer_queue', on_message_callback=self.log)
        self.channel.basic_consume(
            queue='prediction_queue', on_message_callback=self.receive_predict)
        self.channel.basic_consume(
            queue='location_queue', on_message_callback=self.receive_location)
        # bắt đầu lắng nghe tin nhắn từ hàng đợi rpc_queue, nếu có tin nhắn thì gọi đến hàm on_request rồi truyền vào các tham số
        # hình dung hàng đợi này là 1 cái kênh và các client sẽ gửi tin nhắn đến đây, rồi on_request sẽ xử lý tin nhắn đó

        self.data = config["data"]  # truy cập vào data trong file cấu hình
        # truy cập vào debug-mode trong file cấu hình
        self.debug_mode = config["debug-mode"]

        # truy cập vào log-path trong file cấu hình
        log_path = config["log-path"]
        # tạo ra 1 logger mới từ class logger với đường dẫn là log_path + app.log
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        # nằm trong class logger
        self.logger.log_info(
            f"Application start. Server is waiting for {self.total_clients} clients.")
        # đây là 1 phương thức trong class logger, in ra thông báo là server đang chờ đợi các client kết nối đến

    def add_prediction(self, prediction):
        self.predictions.append(prediction)

    def add_processed_prediction(self, prediction):
        self.Processed_predictions.append(prediction)

    def add_prediction_and_location(self, prediction, location):
        self.predictionAndlocation.append((prediction, location))

    def receive_location(self, ch, method, props, body):
        data = json.loads(body)
        kinh_do = data.get('lng')
        vi_do = data.get('lat')
        time = data.get('timestamp')
        location = (kinh_do, vi_do, time)
        while True:
            if len(self.Processed_predictions) > 0:
                self.add_prediction_and_location(
                    self.Processed_predictions, location)
                a, b, c = self.predictionAndlocation[0][1]
                d = [pred.tolist()
                     for pred in self.predictionAndlocation[0][0]]
                print("dự đoán", d[0][0][5])
                data = {
                    'prediction': d[0][0][5],
                    'kinhdo': a,
                    'vido': b,
                    'timestamp': c
                }
                self.channel.basic_publish(exchange='',  # chỉ là gửi tin nhắn đến hàng đợi trên thôi
                                           routing_key='prediction_and_location',
                                           body=json.dumps(
                                               data).encode('utf-8'),
                                           properties=pika.BasicProperties(content_type='application/json'))
                break
        print("Location received:", location)
        print("prediction_and_location:", self.predictionAndlocation)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def receive_predict(self, ch, method, props, body):
        packet = pickle.loads(body)
        # tail gửi về (predictions, meta)
        preds, meta = (packet if isinstance(packet, tuple)
                       and len(packet) == 2 else (packet, None))

        if not isinstance(preds, (list, tuple)) or len(preds) < 2:
            print(
                f"- preds không phải list>=2, type={type(preds).__name__}, content={preds}")
            self.add_processed_prediction(self.predictions[0])
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print("====================================")
            return
        # lấy phần tử đầu tiên trong preds đây mới là tensor dự đoán thô
        p0 = preds[0] if isinstance(preds, (list, tuple)) else preds
        if torch.is_tensor(p0) and p0.ndim == 3:
            print(
                f"- Prediction thô trước khi qua NMS: ={p0.shape}, dtype={p0.dtype}, device={p0.device}")
            display_top_5_scores(p0)
            with torch.no_grad():
                nms_out = ops.non_max_suppression(
                    prediction=p0,
                    conf_thres=0.20,  # nếu conf của box <0.25 thì bỏ qua không xét đến vì chỉ xét conf từ đó lên
                    iou_thres=0.45,  # nếu iou giữa 2 box >0.45 thì coi như trùng box nên bỏ bớt 1 box đi
                    classes=None,
                    agnostic=False,
                    multi_label=False,
                    max_det=300,
                    nc=4,
                )
            det = nms_out[0]  # lấy phần tử đầu tiên trong danh sách nms_out
            # đầu ra là tensor 2D [M, 6], M là số box còn lại sau NMS, 6 là 4 tọa độ box + conf + class
            # scale_boxes chính xác
            if det is not None and len(det) and meta is not None:
                im_shape = meta["im_shape"]      # (1,3,h,w)
                orig_shape = meta["orig_shape"]  # (H,W,3)
                det[:, :4] = ops.scale_boxes(
                    im_shape[2:], det[:, :4], orig_shape).round()
                det[:, :4].clamp_(min=0)

            print(
                f"- Tensor sau khi đi qua NMS: ={det.shape}, dtype={det.dtype}, device={det.device}")
            self.add_prediction(det)
            for i in range(det.shape[0]):
                print(
                    f"- Giá trị xyxy của box {i+1}: {det[i, :6].tolist()}")
                print(
                    f"- Giá trị conf của box {i+1}: {det[i, 4].item():.6f}")
                print(
                    f"- Id class của box {i+1}: {det[i, 5].item():.6f}")

            # # vẽ hộp bao quanh các đối tượng được phát hiện và lưu thành 1 bản sao
            CUR_DIR = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(CUR_DIR, "img1.jpg")
            orig_img = cv2.imread(image_path)

            if orig_img is None:
                print("Không đọc được ảnh: img1.jpg")
            if orig_img.ndim != 3 or orig_img.shape[2] != 3:
                print("Ảnh phải là H×W×3 (BGR)")
            if orig_img.dtype != np.uint8:
                orig_img = orig_img.astype(np.uint8)

            draw = orig_img.copy()
            dummy_img = torch.zeros(
                (1, 3, 640, 640), dtype=p0.dtype, device=p0.device)
            if det is not None and len(det):
                det[:, :4] = ops.scale_boxes(
                    dummy_img.shape[2:], det[:, :4], draw.shape).round()
                xyxy = det[:, :4].clamp(min=0).cpu().numpy().astype(int)
                confs = det[:, 4].cpu().numpy()
                clss = det[:, 5].cpu().numpy().astype(int)
                for (x1, y1, x2, y2), sc, k in zip(xyxy, confs, clss):
                    label = f"{k} {sc:.2f}"
                    color = (0, 0, 255)
                    cv2.rectangle(draw, (x1, y1), (x2, y2), color, 10)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 3, 1)
                    y0 = max(0, y1 - th - 6)
                    cv2.rectangle(draw, (x1, y0), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(draw, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite('output.jpg', draw)
        else:
            print(
                f"- Prediction không phải Tensor 3D như kỳ vọng. type={type(p0).__name__}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    # mục đích là xử lý yêu cầu từ client gửi đến server, server sẽ đăng ký client
    def on_request(self, ch, method, props, body):
        # và kiểm tra số lượng đã đủ chưa
        # tham số của 1 yêu cầu: ch là kênh, methosd là phương thức gửi, properties là thuộc tính, body là nội dung tin nhắn
        # tất cả cái này là thông tin từ client đưa đến nằm trong hàng đợi rpc_queue
        message = pickle.loads(body)
        action = message["action"]
        # đây là id của cái client chạy 1 phần của model
        client_id = message["client_id"]
        # id của layer khi chia model thành 3 layer
        layer_id = message["layer_id"]

        if action == "REGISTER":
            # id của client với id của layer lớn chia từ model mà không có
            if (str(client_id), layer_id) not in self.list_clients:
                # thì thêm client đó vào danh sách clients gồm đki với k đki
                self.list_clients.append((str(client_id), layer_id))
            # in ra tin nhắn từ client
            src.Log.print_with_color(
                f"[<<<] Received message from client: {message}", "blue")
            # tức client chạy phần đầu là 1 thì lưu vào list thì lấy chỉ số 0
            self.register_clients[layer_id-1] += 1

            if self.register_clients == self.total_clients:  # kiểm tra client đăng ký đủ số lượng client ban đầu
                src.Log.print_with_color(
                    "All clients are connected. Sending notifications.", "green")
                # nếu như đủ số lượng client thì gửi thông tin đến tất cả client để bắt đầu
                self.notify_clients()
        else:
            timer = message["timer"]
            src.Log.print_with_color(f"Time: {timer}", "green")

        # trong method có delivery_tag, đây là 1 cái id để xác nhận cái tin nhắn
        ch.basic_ack(delivery_tag=method.delivery_tag)
        # trong hàng đợi, đưa tin nhắn đó vào hàm basic_ack để xác nhận đã nhận được tin nhắn đó rồi là ok

    # mục đích là xử lý yêu cầu từ client gửi đến server, server sẽ đăng ký client
    def log(self, ch, method, props, body):
        # và kiểm tra số lượng đã đủ chưa
        # tham số của 1 yêu cầu: ch là kênh, methosd là phương thức gửi, properties là thuộc tính, body là nội dung tin nhắn
        # tất cả cái này là thông tin từ client đưa đến nằm trong hàng đợi rpc_queue
        message = pickle.loads(body)
        timer = message["timer"]
        layer_id = message["layer_id"]
        src.Log.print_with_color(
            f"Time: {timer} of layer_id: {layer_id}", "green")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_response(self, client_id, message):
        # trả lời lên hàng đợi của riêng client đó
        reply_queue_name = f"reply_{client_id}"

        self.reply_channel.queue_declare(
            reply_queue_name, durable=False)  # khai báo 1 hàng đợi mới để
        # trả lời lại với tên hàng đợi là tên của client, không bền vững vì durable=False
        src.Log.print_with_color(
            # thông báo
            f"[>>>] Sent notification to client {client_id}", "red")

        self.reply_channel.basic_publish(  # xuất vào hàng đợi đó, gồm key và tin nhắn
            exchange='',
            # phải gửi đến đúng tên hàng đợi của client đã khai báo ở trên
            routing_key=reply_queue_name,
            body=message
        )

    def start(self):
        # bắt đầu tiêu thụ hay lắng nghe các tin nhắn trong hàng đợi rpc_queue đối với server
        self.channel.start_consuming()

    def notify_clients(self):  # gửi các thông tin quan trọng đến cho các client

        default_splits = {  # đây chỉ là 1 cái mặc định thôi, có thể thay đổi tùy theo ý muốn
            "a": (1, 9),
            "b": (9, 16),
            "c": (16, 22)
        }

        # ở đây cut_layer đang là a, tức lấy hàng đầu, lấy thông tin các điểm cắt
        splits = default_splits[self.cut_module]

        file_path = f"{self.model_name}.pt"
        if os.path.exists(file_path):  # nếu file đó tồn tại
            # cái này chỉ là ỉn ra thôi
            src.Log.print_with_color(f"Load model {self.model_name}.", "green")
            # rb là read binary, with chỉ là file tự đóng khi kết thúc
            with open(f"{self.model_name}.pt", "rb") as f:
                # hiểu đơn giản f chính là file đó
                file_bytes = f.read()  # đọc file, file này chính là chuỗi byte
                encoded = base64.b64encode(file_bytes).decode('utf-8')
            # từ file .pt mở ra dạng byte sau đó mã hóa chuyển thành dạng base64 sau đó giải mã chuyển thành utf-8 dễ đọc hơn
        else:  # nếu file không tồn tại, thì thông báo rồi thoát
            src.Log.print_with_color(
                f"{self.model_name} does not exist.", "yellow")
            sys.exit()

        for (client_id, _) in self.list_clients:

            response = {"action": "START",
                        "message": "Server accept the connection",
                        "model": encoded,  # lấy file model yolov8n đã mã hóa ở trên rồi truyền vào các client
                        "split_module1": splits[0],
                        "split_module2": splits[1],
                        # chính là 1 là mỗi lần xử lý cùng lúc bao nhiêu khung hình
                        "batch_frame": self.batch_frame,
                        # số lượng layer trong model chính bằng với số lượng client, vì chia thành 2 layer
                        "num_layers": len(self.total_clients),
                        # hiểu là gom toàn bộ layer con thành 1 layer lớn, thì đây là bao nhiêu layer lớn
                        "model_name": self.model_name,  # chính là tên model
                        # chính là file video cần nhận diện và mình sẽ đưa nó đến cho client đầu nhận rồi chạy inference
                        "data": self.data,
                        "debug_mode": self.debug_mode}  # chế độ gỡ lỗi

            self.send_to_response(client_id, pickle.dumps(response))

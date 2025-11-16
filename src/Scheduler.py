import base64
import os
import pickle
import time
from tqdm import tqdm
import torch
import cv2
import pika
from src.Model import SplitDetectionPredictor
import numpy as np
import json
from ultralytics.utils import ops


def decode_image_base64_to_cv2(image_b64: str):
    """Giải mã base64 -> OpenCV image (BGR)."""
    img_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    return img


def letterbox_img(img_rgb, new_shape=640, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img_rgb.shape[:2]  # (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])
        dw, dh = 0, 0

    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img_resized = cv2.resize(
            img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_rgb

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    ratio = (r, r)
    return img_padded, ratio, (dw, dh)


class Scheduler:  # mục đích là tính thời gian inference trên từng phần head, mid, tail, sắp xếp giao tiếp các client với nhau và với server
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id  # id của client
        # id chỉ vị trí của client là nằm đầu hay sau hay giữa của model
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        # hàng đợi này là của cái layer_id sẽ gửi dữ liệu đến đây
        self.intermediate_queue = f"intermediate_queue_{self.layer_id}"
        # chính là khai báo 1 hàng đợi dành riêng cho từng layer_id
        self.channel.queue_declare(self.intermediate_queue, durable=False)
        # durable = false có nghĩa là sẽ không lưu lại tin nhắn trong hàng đợi, mà tin nhắn gửi đến bên kia sẽ dùng ngay

    # mục đích là gửi dữ liệu đến layer tiếp theo

    def send_next_part(self, intermediate_queue, data, logger):
        if data != 'STOP':  # tức là chưa dừng lại thì
            data["modules_output"] = [t.cpu() if isinstance(
                t, torch.Tensor) else None for t in data["modules_output"]]
            # data tại vị trí layers_output chứa ds các đầu ra của module quan trọng, còn không quan trọng thì là None
            # nó duyệt từng module quan trọng trong layers_output, nếu là tensor thì chuyển về cpu, còn không thì cho là None
            # sau đó lưu lại vào data ở vị trí layers_output
            message = pickle.dumps({  # chính là đóng gói lại dữ liệu để gửi đi layer khác rồi lưu thành message, dữ liệu
                # sẽ gồm các đầu ra của module quan trọng
                "action": "OUTPUT",
                "data": data
            })

            self.channel.basic_publish(  # mục đích chính là xuất vào hàng đợi ở trên, layer khác sẽ vào hang đợi này để lấy dữ liệu
                exchange='',
                routing_key=intermediate_queue,  # đây là tên hàng đợi ứng với từng layer_id
                body=message,
            )
        else:  # còn nếu nhận được tin nhắn dừng lại thì
            message = pickle.dumps(data)  # vẫn đóng gói dữ liệu lần cuối
            self.channel.basic_publish(  # sau đó lại xuất vào hàng đợi đó tiếp, tương ứng với layer_id
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )

    # mục đích là lấy video đầu vào, rồi cho chạy qua head

    def inference_head(self, model, data, batch_frame, logger):

        time_inference = 0  # thời gian inference
        i = 1
        count = 0
        similarity_tensor = []
        # đây là lớp đưa đầu ra của model vào để xử lý
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        model.eval()
        model.to(self.device)
        self.channel.queue_declare(queue='image_queue', durable=True)
        self.channel.basic_qos(prefetch_count=50)
        while True:
            method_frame, _, body = self.channel.basic_get(
                queue='image_queue', auto_ack=True)
            if body:
                payload = json.loads(body.decode('utf-8'))
                img_bgr = decode_image_base64_to_cv2(
                    payload['image_base64'])
                if img_bgr is None:
                    print("Không decode được ảnh.")
                    self.channel.basic_ack(
                        delivery_tag=method_frame.delivery_tag)
                    return
                elif img_bgr is not None:
                    print("shape:", img_bgr.shape)
                img0 = img_bgr
                img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                # LetterBox giống Ultralytics
                img_lb, ratio, (dw, dh) = letterbox_img(
                    img_rgb, new_shape=640, auto=True, scaleFill=False, scaleup=True, stride=32
                )

                im = img_lb.transpose((2, 0, 1)).astype(
                    np.float32) / 255.0  # CHW
                im = np.ascontiguousarray(im)
                im = torch.from_numpy(im).unsqueeze(
                    0).to(self.device)        # (1,3,h,w)

                # Lưu thông tin scale để tail dùng
                meta = {
                    "im_shape": im.shape,
                    "orig_shape": img0.shape,   # BGR gốc
                    "ratio": ratio,
                    "pad": (dw, dh),
                }
                # frame = cv2.resize(img_bgr, (640, 640))
                # tensor = torch.from_numpy(frame).float().permute(2, 0, 1)
                # tensor /= 255.0
                # list_average_tensor = []
                # list_average_tensor.append(tensor)
                # reference_tensor = torch.stack(list_average_tensor)
                # reference_tensor = reference_tensor.to(self.device)
                # predictor.setup_source(reference_tensor)
                # for predictor.batch in predictor.dataset:
                #     path, average_tensor, _ = predictor.batch
                # preprocess_image = predictor.preprocess(average_tensor)

                start = time.time()
                # # chạy trên phần head
                # # kết quả chính là dạng key-value
                y = model.forward_head(im)
                time_inference += (time.time() - start)

                # gửi kèm meta qua queue
                y["meta"] = meta
                self.send_next_part(self.intermediate_queue, y, logger)

                y = 'STOP'
                self.send_next_part(self.intermediate_queue, y, logger)
                logger.log_info(f"End Inference Head.")
                return time_inference
            else:
                continue
        # liên tục nhận dữ liệu từ web xuống
        # path = None
        # data_path = data
        # data_path_extension = os.path.splitext(data_path)[1].lower()

        # if data_path_extension == '.jpg':
        #     img = cv2.imread(data_path)  # đọc ảnh từ đường dẫn
        #     img = cv2.resize(img, (640, 640))
        #     # chuyển từ mảng thành tensor nhưng lại có dạng (0-255),
        #     tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        #     # trong khi pytoch cần float (0.0-255.0) nên cần chuyển thành float, chuyển thứ tự height, width, kênh màu thành
        #     #  kênh màu, height, width, nên sẽ có shape là (3, 640, 640)
        #     tensor /= 255.0  # mỗi pixel có giá trị 0.0-255.0 nên chia 255.0 để đưa về khoảng 0.0-1.0
        #     list_average_tensor = []
        #     list_average_tensor.append(tensor)
        #     reference_tensor = torch.stack(list_average_tensor)
        #     # chuyển tensor đến device để tính toán, cụ thể ở đây là cuda
        #     reference_tensor = reference_tensor.to(self.device)
        #     # cuda chính là nền tảng tính toán song song trên GPU do NVIDIA phát triển
        #     predictor.setup_source(reference_tensor)
        #     for predictor.batch in predictor.dataset:
        #         path, average_tensor, _ = predictor.batch
        #     preprocess_image = predictor.preprocess(average_tensor)
        #     start = time.time()
        #     y = model.forward_head(preprocess_image)
        #     time_inference += (time.time() - start)
        #     self.send_next_part(self.intermediate_queue, y, logger)
        #     y = 'STOP'
        #     self.send_next_part(self.intermediate_queue, y, logger)
        #     logger.log_info(f"End Inference Head.")
        #     return time_inference

        # elif data_path_extension == '.mp4':
        #     # cap chính là cái video sau khi mở ra từng path
        #     cap = cv2.VideoCapture(data_path)
        #     if not cap.isOpened():
        #         logger.log_error(f"Not open video")
        #         return False
        #     # frame per second, chính là số khung hình của video lấy trong 1s
        #     fps = cap.get(cv2.CAP_PROP_FPS)
        #     logger.log_info(f"FPS input: {fps}")
        #     # sẽ hiển thị ra cho mình xem % hoàn thành video
        #     pbar = tqdm(desc="Processing video (while loop)", unit="frame")

        #     # if self.layer_id==1:
        #     #     timer = time.time()
        #     #     timer = {"timer": timer, "layer_id":1}
        #     #     self.connect()
        #     #     self.channel.queue_declare('timer_queue', durable=False) # chính là cái hàng đợi gửi lên để bên server lắng nghe
        #     #     self.channel.basic_publish(exchange='', # chỉ là gửi tin nhắn đến hàng đợi trên thôi
        #     #                         routing_key='timer_queue',
        #     #                         body=pickle.dumps(timer))
        #     while True:  # vòng lặp này là để lấy từng khung hình trong video lặp cho đến hết video, hiểu từng khung hình là từng cái hình thôi
        #         # còn video là lắp ghép của rất nhiều hình lại
        #         start = time.time()

        #         # ret là return nếu trả về true tức khung frame được đọc thành công
        #         ret, frame = cap.read()
        #         # còn frame sẽ là mảng 3 chiều lưu tính chất của từng pixel gồm height, width và màu

        #         if not ret and reference_tensor is not None:  # nếu không đọc được
        #             list_reference_tensor = []
        #             list_reference_tensor.append(reference_tensor)
        #             reference_tensor = torch.stack(list_reference_tensor)
        #             reference_tensor = reference_tensor.to(self.device)
        #             predictor.setup_source(reference_tensor)
        #             for predictor.batch in predictor.dataset:
        #                 path, reference_tensor, _ = predictor.batch
        #             preprocess_image = predictor.preprocess(reference_tensor)
        #             y = model.forward_head(preprocess_image)
        #             time_inference += (time.time() - start)
        #             self.send_next_part(self.intermediate_queue, y, logger)
        #             reference_tensor = None
        #             pbar.update(batch_frame)

        #         if not ret and reference_tensor is None:  # nếu không đọc được
        #             y = 'STOP'
        #             # thì gửi tin nhắn stop lên hàng đợi để layer kia biết là
        #             self.send_next_part(self.intermediate_queue, y, logger)
        #             # không có video, sau đó kết thúc
        #             break
        #         # đây chính là mảng 3 chiều lưu các pixel của từng khung hình trong video
        #         frame = cv2.resize(frame, (640, 640))
        #         # bây giờ thay đổi đúng kích thước đầu vào
        #         # chuyển từ mảng thành tensor nhưng lại có dạng (0-255),
        #         tensor = torch.from_numpy(frame).float().permute(2, 0, 1)
        #         # trong khi pytoch cần float (0.0-255.0) nên cần chuyển thành float, chuyển thứ tự height, width, kênh màu thành
        #         #  kênh màu, height, width, nên sẽ có shape là (3, 640, 640)
        #         tensor /= 255.0  # mỗi pixel có giá trị 0.0-255.0 nên chia 255.0 để đưa về khoảng 0.0-1.0
        #         if i == 1:
        #             reference_tensor = tensor
        #             i += 1
        #             continue  # bỏ qua lần đầu tiên vì chưa có tensor để so sánh
        #         similarity = torch.nn.functional.cosine_similarity(
        #             tensor.flatten(), reference_tensor.flatten(), dim=0)  # so sánh
        #         count += 1
        #         logger.log_info(f"similarity: {similarity} - {count}")

        #         if similarity > 0.99:  # nếu giống nhau quá
        #             continue  # nếu giống nhau quá thì bỏ qua không cần truyền đi

        #         else:
        #             list_average_tensor = []
        #             list_average_tensor.append(reference_tensor)
        #             reference_tensor = torch.stack(list_average_tensor)
        #             # chuyển tensor đến device để tính toán, cụ thể ở đây là cuda
        #             reference_tensor = reference_tensor.to(self.device)
        #             # cuda chính là nền tảng tính toán song song trên GPU do NVIDIA phát triển
        #             # chuẩn bị dữ liệu
        #             # logger.log_info(f"---------average: {average_tensor.shape} ---------")
        #             # chuẩn bị dữ liệu cho model, cụ thể là đầu vào của model
        #             predictor.setup_source(reference_tensor)
        #             # hàm này sẽ đọc dữ liệu đầu vào, sau đó tiền xử lý resize về 640x640 như yêu cầu của tham số đầu vào của predictor
        #             # mỗi batch hay mỗi lô hàng trong dataset chính là 1 khung hình, vì input_image
        #             for predictor.batch in predictor.dataset:
        #                 # là lưu nhiều khung hình, vì nó sẽ lặp qua các khung hình rồi lưu vào input_image, và 1 lần sẽ xử lý cùng lúc input_image
        #                 # với số lượng là batch_frame
        #                 # mỗi bacth hay lô hàng chính là 1 khung hình sẽ có path, các pixel ở dạng tensor
        #                 path, average_tensor, _ = predictor.batch
        #             # và các pixel ở dạng mảng numpy
        #             # kết quả sẽ được: input_image sẽ lưu các pixel dạng tensor của từng khung hình cần xử lý cùng lúc, nó sẽ là list

        #             # tiền xử lý ảnh sau khi input_image đã lưu các pixel của từng khung hình
        #             # đưa các pixel đó vào tiền xử lý sẽ được các tensor của từng khung
        #             preprocess_image = predictor.preprocess(average_tensor)
        #             # quá trình này sẽ gồm resize, chuẩn hóa pixel, chuyển thành kênh màu, height, width

        #             # ảnh sau khi xử lý xong sẽ đưa qua model ở phần head trước
        #             # save_layers chính là các module quan trọng cần lưu lại đầu ra
        #             y = model.forward_head(preprocess_image)
        #             # kết quả sẽ được y là dạng key-value, layers_output sẽ là 1 list chứa các đầu ra của các module quan trọng, còn lại None
        #             # còn last_layer_idx chính là vị trí cuối cùng trong ds y, tức là module cuối cùng trong phần head

        #             # tính thời gian inference cho mỗi batch_frame, chính là cùng lúc số khung hình
        #             time_inference += (time.time() - start)
        #             # cái này là cộng lại để lấy tổng thời gian là bao nhiêu
        #             reference_tensor = tensor
        #             # if self.layer_id==1:
        #             #     timer = time.time()
        #             #     timer = {"timer": timer, "layer_id":1}
        #             #     self.connect()
        #             #     self.channel.queue_declare('timer_queue', durable=False) # chính là cái hàng đợi gửi lên để bên server lắng nghe
        #             #     self.channel.basic_publish(exchange='', # chỉ là gửi tin nhắn đến hàng đợi trên thôi
        #             #             routing_key='timer_queue',
        #             #             body=pickle.dumps(timer))

        #             # đưa dữ liệu đầu ra từ phần head lên hàng đợi
        #             self.send_next_part(self.intermediate_queue, y, logger)
        #             # để phần mid lên đó lấy dữ liệu để chạy inference tiếp
        #             # cập nhật lại thanh tiến độ, tức là đã xử lý xong batch_frame khung hình rồi, nó sẽ cộng vào
        #             pbar.update(batch_frame)
        #             # sẽ hiển thị % hoàn thành theo thời gian thực

        #     cap.release()
        #     pbar.close()
        #     logger.log_info(f"End Inference Head.")
        #     return time_inference
        # else:
        #     logger.log_error(f"Not support data format: {data_path_extension}")
        #     return False

    def inference_mid(self, model, batch_frame, logger):
        time_inference = 0

        model.eval()
        model.to(self.device)
        # - 1 có nghĩa là nếu đang 2 tức nó là phần giữa model muốn lấy phần đầu thì phải -1
        last_queue = f"intermediate_queue_{self.layer_id - 1}"
        # khai báo tạo ra hàng đợi
        self.channel.queue_declare(queue=last_queue, durable=False)
        self.channel.basic_qos(prefetch_count=50)

        pbar = tqdm(desc="Processing video (while loop)", unit="frame")

        while True:
            method_frame, _, body = self.channel.basic_get(
                queue=last_queue, auto_ack=True)  # truy cập vào hàng đợi phía trên
            # để lấy tin nhắn, chính là đầu ra của từ các module ở phần head, liên tục lấy tin nhắn về vì có vòng while
            if method_frame and body:

                received_data = pickle.loads(body)
                if received_data != 'STOP':
                    y = received_data["data"]  # chính là đầu ra dạng key-value
                    meta = y.get("meta", None)  # giữ lại
                    # key này chính là đầu ra
                    y["modules_output"] = [
                        t.to(self.device) if t is not None else None for t in y["modules_output"]]
                    start = time.time()
                    # chạy trên phần mid
                    y = model.forward_mid(y)  # kết quả chính là dạng key-value
                    if "meta" in received_data["data"]:
                        y["meta"] = received_data["data"]["meta"]

                    time_inference += (time.time() - start)

                    self.send_next_part(self.intermediate_queue, y, logger)

                    pbar.update(batch_frame)  # cập nhật rồi hiển thị thanh %
                else:
                    break
            else:
                continue

        y = 'STOP'
        self.send_next_part(self.intermediate_queue, y, logger)

        pbar.close()
        logger.log_info(f"End Inference Mid.")
        return time_inference

    # không có data và save_layers ở đây, vì đến cuối rồi, nó chỉ lấy dữ liệu từ hàng đợi thôi
    def inference_tail(self, model, batch_frame, logger):
        time_inference = 0

        model.eval()
        model.to(self.device)
        last_queue = f"intermediate_queue_{self.layer_id - 1}"
        # khai báo tạo ra hàng đợi
        self.channel.queue_declare(queue=last_queue, durable=False)
        # qos là quality of service, số lượng message chưa ack tối đa mà consumer có thể xử lý,
        self.channel.basic_qos(prefetch_count=50)
        # ở đây message chưa ack - acknowledge (hiểu đơn giản là chưa được xác nhận đã xử lý xong, nghĩa là message vẫn đang xử lý
        # trên consumer đó)
        # tiếp tục sẽ là thanh để hiển thị % hoàn thành video
        pbar = tqdm(desc="Processing video (while loop)", unit="frame")

        while True:
            method_frame, header_frame, body = self.channel.basic_get(
                queue=last_queue, auto_ack=True)  # truy cập vào hàng đợi phía trên
            # để lấy tin nhắn, chính là đầu ra của từ các module ở phần head
            if method_frame and body:

                received_data = pickle.loads(body)
                if received_data != 'STOP':
                    y = received_data["data"]  # chính là đầu ra dạng key-value
                    meta = y.get("meta", None)
                    # key này chính là đầu ra
                    y["modules_output"] = [
                        t.to(self.device) if t is not None else None for t in y["modules_output"]]
                    start = time.time()
                    # if self.layer_id==3:
                    #     timer = time.time()
                    #     timer = {"timer": timer, "layer_id":3}
                    #     self.connect()
                    #     self.channel.queue_declare('timer_queue', durable=False) # chính là cái hàng đợi gửi lên để bên server lắng nghe
                    #     self.channel.basic_publish(exchange='', # chỉ là gửi tin nhắn đến hàng đợi trên thôi
                    #             routing_key='timer_queue',
                    #             body=pickle.dumps(timer))
                    # chạy trên phần tail
                    # kết quả chính là tensor đầu ra của model
                    predictions = model.forward_tail(y)
                    meta = y.get("meta", None)
                    # cái dự đoán này là cho từng frame tức là dự đoán cho từng khung hình trong video và nó có 3 scale khác nhau, chứ không phải cả video

                    # chạy hậu xử lý từ kết quả đầu ra của model, tức là xem kết quả đầu ra
                    # if save_output:
                    #     results = predictor.postprocess(predictions, y["img"], y["orig_imgs"], y["path"])
                    # ở trên là sẽ bổ sung thêm mỗi tin nhắn gửi lên hàng đợi ngoài đầu ra của các layer thì cần img, orig_imgs, path để xem kqua

                    time_inference += (time.time() - start)
                    pbar.update(batch_frame)  # cập nhật rồi hiển thị thanh %
                    # vẫn đóng gói dữ liệu lần cuối
                    message = pickle.dumps((predictions, meta))
                    self.channel.basic_publish(
                        exchange='', routing_key='prediction_queue', body=message)
                else:
                    message = pickle.dumps(received_data)
                    self.channel.basic_publish(
                        exchange='', routing_key='prediction_queue', body=message)
                    break
            else:
                continue
        pbar.close()
        logger.log_info(f"End Inference Tail.")
        return time_inference

    def inference_func(self, model, data, num_layers, batch_frame, logger):
        time_inference = 0
        if self.layer_id == 1:  # tức là client nó nằm đầu model
            time_inference = self.inference_head(
                model, data, batch_frame, logger)
        elif self.layer_id == num_layers:  # tức là client nằm ở cuối model
            time_inference = self.inference_tail(model, batch_frame, logger)
        else:
            # nếu không thì nó nằm giữa model
            time_inference = self.inference_mid(model, batch_frame, logger)
        return time_inference

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            self.address, 5672, self.virtual_host, credentials))
        self.channel = self.connection.channel()

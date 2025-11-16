import pickle
import time
import base64
import os

import pika
import torch
import torch.nn as nn

import src.Log
from src.Model import SplitDetectionModel
# tức là phải cài thư viện, file yolo.pt dùng từ thư viện
from ultralytics import YOLO


class RpcClient:  # mục đích là sẽ nhận tin nhắn từ rpc_queue
    def __init__(self, client_id, layer_id, address, username, password, virtual_host, inference_func, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.inference_func = inference_func
        self.device = device

        self.channel = None
        self.connection = None
        self.response = None
        self.model = None
        self.data = None
        self.logger = None
        self.connect()

    def wait_response(self):
        status = True
        reply_queue_name = f"reply_{self.client_id}"
        # đây chính là hàng đợi mà server trả lời tin nhắn vào
        self.channel.queue_declare(reply_queue_name, durable=False)
        # và client sẽ lắng nghe
        while status:
            method_frame, header_frame, body = self.channel.basic_get(
                queue=reply_queue_name, auto_ack=True)  # chính là client sẽ lắng nghe
            # tin nhắn từ hàng đợi rồi lấy tin nhắn về
            if body:  # nếu có tin nhắn từ server
                # thì sẽ phản hồi lại tin nhắn đó và lưu status là false, trước đó
                status = self.response_message(body)
            # thì sẽ xử lý bằng cách client sẽ chia model theo yêu cầu của server, sau đó nó cũng chạy inference luôn
            # và lưu lại thời gian inference
            time.sleep(0.5)

    # phía dưới chính là thực hiện việc chia model theo yêu cầu của server
    def response_message(self, body):
        self.response = pickle.loads(body)  # lưu tin nhắn lại
        src.Log.print_with_color(
            # thông báo
            f"[<<<] Client received: {self.response['message']}", "blue")
        # lấy action từ tin nhắn ra chính là START vì là từ server, còn từ client thì là REGISTER
        action = self.response["action"]

        if action == "START":
            model_name = self.response["model_name"]
            num_layers = self.response["num_layers"]
            split_module1 = self.response["split_module1"]
            split_module2 = self.response["split_module2"]
            batch_frame = self.response["batch_frame"]
            model = self.response["model"]
            data = self.response["data"]
            debug_mode = self.response["debug_mode"]
            self.logger = src.Log.Logger(f"result.log", debug_mode)
            if model is not None:
                file_path = f'{model_name}.pt'
                if os.path.exists(file_path):
                    src.Log.print_with_color(f"Exist {model_name}.pt", "green")
                else:
                    decoder = base64.b64decode(model)
                    with open(f"{model_name}.pt", "wb") as f:
                        f.write(decoder)
                    src.Log.print_with_color(
                        f"Loaded {model_name}.pt", "green")
            else:
                src.Log.print_with_color(f"Do not load model.", "yellow")

            pretrain_model = YOLO(f"{model_name}.pt").model
            self.model = SplitDetectionModel(
                pretrain_model, split_module1=split_module1, split_module2=split_module2)
            # đây chính là lúc thực hiện tách

            start = time.time()
            self.logger.log_info(f"Start Inference")
            # gửi thời gian về server

            if self.layer_id == 1:
                timer = time.time()
                timer = {"timer": timer, "layer_id": 1}
                self.connect()
                # chính là cái hàng đợi gửi lên để bên server lắng nghe
                self.channel.queue_declare('timer_queue', durable=False)
                self.channel.basic_publish(exchange='',  # chỉ là gửi tin nhắn đến hàng đợi trên thôi
                                           routing_key='timer_queue',
                                           body=pickle.dumps(timer))

            time_inference = self.inference_func(
                self.model, data, num_layers, batch_frame, self.logger)

            if self.layer_id == num_layers:
                timer = time.time()
                timer = {"timer": timer, "layer_id": num_layers}
                self.connect()
                # chính là cái hàng đợi gửi lên để bên server lắng nghe
                self.channel.queue_declare('timer_queue', durable=False)
                self.channel.basic_publish(exchange='',  # chỉ là gửi tin nhắn đến hàng đợi trên thôi
                                           routing_key='timer_queue',
                                           body=pickle.dumps(timer))
            # thực hiện việc inference model đã được chia ra, sau đó lấy thời gian inference về
            # cái hàm inference_func này nằm trong class scheduler mà class đó có layer_id rồi nên hàm này sẽ biết tính time inference
            # trên phần nào của model đầu giữa hay cuối
            all_time = time.time() - start
            self.logger.log_info(f"All time: {all_time}s")
            self.logger.log_info(f"Inference time: {time_inference}s")
            self.logger.log_info(
                f"Utilization: {((time_inference / all_time) * 100):.2f} %")

            # Stop or Error
            return False
        else:
            return False

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            self.address, 5672, self.virtual_host, credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        # chính là cái hàng đợi gửi lên để bên server lắng nghe
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',  # chỉ là gửi tin nhắn đến hàng đợi trên thôi
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

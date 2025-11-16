import contextlib
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops


class SplitDetectionModel(nn.Module):
    def __init__(self, cfg, split_module1=-1, split_module2=-1):
        super().__init__()
        self.model = cfg.model
        self.save = cfg.save
        self.stride = cfg.stride
        self.inplace = cfg.inplace
        self.names = cfg.names
        self.yaml = cfg.yaml
        self.nc = len(self.names)
        self.task = cfg.task
        self.pt = True

        if split_module1 > 0 and split_module2 > 0:  # tổng 23 module
            self.head = self.model[:split_module1]  # 0 đến split_module1-1
            # split_module1 đến split_module2-1
            self.mid = self.model[split_module1:split_module2]
            # split_module2 đến 22 (cái 22 là cái nhận 3 đầu ra làm 3 detect, nên 22 nó sẽ có 1 list 3 đầu ra)
            self.tail = self.model[split_module2:]

    def forward_head(self, x):  # x chính là tensor đầu vào layer đầu
        output_from = (4, 6, 9, 12, 15, 18)
        # những chỉ số module nằm trong output_from: 4, 6, 9, 12, 15, 18 thì mới được lưu lại
        # còn lại thì sẽ là None

        y = []  # là danh sách các đầu ra quan trọng cần lưu lại để truyền đến phần mid
        # duyệt qua từng module trong head, lấy chỉ số và module đó
        for _, m in enumerate(self.head):
            if m.f != -1:  # khác -1 là có đầu vào từ 1 hay nhiều module trước
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]
                # mục đích lấy đầu vào cho module hiện tại m
                # nếu m.f nguyên thì lấy chỉ số của layer đó trong mảng y
                # nếu mf không phải nguyên => là 1 danh sách nhiều số nguyên, do là có nhiều đầu vào
                #   lúc đó lấy từng chỉ số j trong danh sách m.f rồi lấy ra từ mảng y lại lưu vào x, vậy x chính là danh sách các module
                # kết quả: được x là những đầu vào của module hiện tại
            # x từ module trước vào module hiện tại, điều này luôn đươc thực hiện vì các module liên tục nhau
            x = m(x)
            # kết quả: được x chính là dữ liệu đầu ra khi tensor qua module hiện tại
            y.append(x if (m.i in self.save) else None)
        if y[-1] is None:
            y[-1] = x
        return {"modules_output": y, "last_modules_idx": len(y) - 1}
        #     if (m.i in self.save) or (i in output_from):
        #         # kiểm tra nếu module hiện tại nằm trong save là ds các module quan trọng cần lưu trữ dữ liệu đầu ra
        #         # hoặc module hiện tại nằm trong ds các module quan trọng mà được chỉ định để lưu trữ dữ liệu đầu ra
        #         # kết quả: lưu lại module vào ds y để sử dụng cho các module sau này
        #         y.append(x)
        #     else:
        #         # nếu không nằm trong ds save và output_from thì tại chỉ số đó lưu lại là None
        #         y.append(None)
        # # sau khi thoát khỏi vòng lặp trên thì sẽ được y chứa các dữ liệu đầu ra theo từng module để truyền đến phần mid
        # # x cuối cùng chính là đầu ra của module cuối cùng trong phần head, cũng nối liền với module đầu tiên trong phần mid

        # for mi in range(len(y)):  # duyệt qua từng module trong y
        #     if mi not in output_from:  # nếu module đó không nằm trong ds các module mình cần lưu dữ liệu đầu ra thì None
        #         y[mi] = None

        # if y[len(y) - 1] is None:
        #     # lưu phần tử cuối bằng đầu ra của module cuối cùng trong head vì nó nối tiếp với module đầu tiên trong mid
        #     y[len(y) - 1] = x

        # return {"modules_output": y, "last_modules_idx": len(y) - 1}
        # kết quả: y là chứa các đầu ra của các module quan trọng, cái không quan trọng thì là None
        # và chỉ số của module cuối cùng ds y, thứ tự đầu đến cuối là cũng từ module 0 đến split_layer1-1

    # y là mảng có các chỉ số là các module quan trọng trong mid và head

    def forward_mid(self, xdict):
        output_from = (4, 6, 9, 12, 15, 18)  # những module quan trọng

        # y bây giờ sẽ giống ds y phần head chứa đầu ra các module quan trọng trong head
        y = xdict["modules_output"]
        x = y[xdict["last_modules_idx"]]
        # last_index = x["last_modules_idx"]
        # x bây giờ chính là đầu ra nằm cuối trong ds y, cũng chính là của module nối liền với phần mid
        # x = y[last_index]

        for _, m in enumerate(self.mid):
            if m.f != -1:  # khác -1 là có đầu vào từ 1 hay nhiều module trước
                # x lưu đầu vào cho module hiện tại
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]
            # cho đầu vào x vào module hiện tại, điều này luôn được thực hiện vì các module liên tục nhau
            x = m(x)
            # nếu module hiện tại quan trọng thì lưu lại
            # if (m.i in self.save) or (i in output_from):
            #     y.append(x)
            # else:
            #     y.append(None)
            y.append(x if (m.i in self.save) else None)
        if y[-1] is None:
            y[-1] = x
        return {"modules_output": y, "last_modules_idx": len(y) - 1}

        # for mi in range(len(y)):
        #     if mi not in output_from:  # nếu module đó không quan trọng thì cho thành None
        #         y[mi] = None

        # # kiểm tra phần tử cuối cùng rồi lưu lại bằng đầu ra của module cuối cùng trong mid
        # if y[len(y) - 1] is None:
        #     y[len(y) - 1] = x
        # return {"modules_output": y, "last_modules_idx": len(y) - 1}

    # x chính là đầu vào, x=model.forward_mid(x), qua tail cũng chính là tensor luôn
    def forward_tail(self, x):

        # y bây giờ sẽ giống ds y chứa đầu ra các module trong mid
        y = x["modules_output"]
        last_index = x["last_modules_idx"]
        # x bây giờ chính là đầu ra nằm cuối trong ds y, cũng chính là của module nối liền với phần tail này
        x = y[last_index]

        for m in self.tail:
            if m.f != -1:  # khác -1 là có đầu vào từ 1 hay nhiều module trước
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)

        # kết thúc toàn bộ for thì cuối cùng chỉ có 1 đầu ra duy nhất là x và lưu vào y, đầu ra đó là list gồm 3 phần tử
        y = x  # y  là list hoặc tuple, lí do là vì module cuối cùng của model nó có 3 đầu ra chính là 3 scale nên đầu ra phải là list
        # hoặc đã được concat thành 1 tensor
        if isinstance(y, (list, tuple)):
            # chuyển thành tensor hoặc list tensor
            return self.from_numpy(y[0] if len(y) == 1 else [self.from_numpy(x) for x in y])
        else:
            return self.from_numpy(y)

    def _predict_once(self, x):  # cách làm này để kiểm tra cả cái model ban đầu, không chia
        y = []  # outputs
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x  # đầu ra cuối cùng của model chính là x đây là trong trường hợp không chia model

    def forward(self, x):  # cho chạy bằng model ban đầu
        return self._predict_once(x)

    def from_numpy(self, x):
        # nếu x là numpy array chính là
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x
        # đầu ra cuối cùng từ tail thì chuyển thành tensor sau đó chuyển về device của model, còn không thì trả về x luôn


# cái class này nó kế thừa từ DetectionPredictor tức dự đoán cho model phát hiện
class SplitDetectionPredictor(DetectionPredictor):
    # chức năng của cha nó bao gồm setup_source (chuẩn bị nguồn dữ liệu), preprocess (tiền xử lý), dataset để duyệt theo từng batch
    # THỨ TỰ TRƯỚC KHI ĐƯA VÀO MODEL LÀ:
    # img -> tensor -> predictor.setup_source -> predictor.dataset -> predictor.batch-> predictor.preprocess() -> model -> postprocess
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        model.fp16 = self.args.half
        self.model = model

    def postprocess(self, preds, img, orig_imgs=None, path=None):
        preds = ops.non_max_suppression(preds,  # predictions thô từ model đưa vào NMS
                                        self.args.conf,  # ngưỡng tin cậy để loại bỏ các dections không chắc chắn
                                        self.args.iou_thres,
                                        self.args.iou,  # ngưỡng IOU để loại bỏ các dections trùng lặp
                                        self.args.classes,  # lớp nào được phép detect, chỉ giữ lại các lớp này như người/ vật
                                        self.args.agnostic_nms,  # tối ưu khi detect nhiều lớp khác nhau cùng 1 lúc
                                        max_det=self.args.max_det,  # số lượng ảnh tối đa detect được, ngăn tràn bộ nhớ
                                        # kích thước của tensor output đúng với số lớp
                                        nc=len(self.model.names),
                                        )
        if orig_imgs is not None and not isinstance(orig_imgs, list):
            # chuyển ảnh gốc từ tensor về numpy array
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # lấy predictions sau khi loại bỏ bằng non_max_suppression
        return self.construct_results(preds, img, orig_imgs, path)
        # trả về kết quả cuối cùng là 1 list chứa các kết quả của từng ảnh trong batch

    def construct_results(self, preds, img, orig_imgs, path):
        return [
            # đưa từng đối tượng trong preds, img, orig_img, img_path
            self.construct_result(pred, img, orig_img, img_path)
            # lấy từng đối tuợng trong preds, orig_imgs, path
            for pred, orig_img, img_path in zip(preds, orig_imgs, path)
        ]

    def construct_result(self, pred, img, orig_img, img_path):  # các đối tượng đơn đưa vào
        # pred có dạng [N,6] với N là số lượng bounding box, 6 là cột [x1,y1,x2,y2,conf,cls]
        # nên [:, :4] là tất cả các box và 4 cột đầu tiên chỉ số 0-3,
        #     [:, 4:6] là tất cả các box và 2 cột sau chỉ sô 4-5
        # chia tỉ lệ ảnh gốc về kích thước ảnh đầu vào
        if pred is not None and len(pred):
            pred[:, :4] = ops.scale_boxes(
                img.shape[2:], pred[:, :4], orig_img.shape)
        # lí do là vì các bounding box trong pred là dựa vào kích thước ảnh đầu vào (ảnh đó đã tiền xử lý nên đã thay đổi kích thước),
        # còn ảnh gốc kích thước khác nên phải chia tỉ lệ lại cho đúng với ảnh đầu vào, để vẽ bounding box lên ảnh gốc cho đúng vị trí
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6] if pred is not None else pred)

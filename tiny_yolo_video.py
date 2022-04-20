import cv2
import quadric
import numpy as np
import sys
import os
import io
import json
from typing import Optional, Tuple


labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

class NmsOutputShapes:
    def __init__(self, max_allowed_boxes=160, num_classes=80):

        self.vboxes_shape = (1, 1, 4, max_allowed_boxes)
        self.vboxes_dtype = np.dtype(np.int32)

        self.vbox_ids_shape = (
            1, 1, max_allowed_boxes, num_classes)
        self.vbox_ids_dtype = np.dtype(np.int16)

        self.vbox_count_shape = (1, 1, 1, num_classes)
        self.vbox_count_dtype = np.dtype(np.int32)

        self.score_shape = (
            1, 1, max_allowed_boxes, num_classes)
        self.score_dtype = np.dtype(np.int16)

class CameraStreamInput:
    """
    Initializes a camera stream and returns it as an iterable object
    """

    def __init__(self):
        self.vid = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._index = 0

    def __iter__(self):
        """
        Creates an iterator for this container.
        """
        self._index = 0
        return self

    def __next__(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        @return tuple containing current image and meta data if available, otherwise None
        """
        success, frame = self.vid.read()
        if(success):
            self._index += 1
            return (frame, self._index)
        else:
            raise StopIteration


def draw_boxes(image, vboxes, vbox_count, vbox_ids, hscale=1, vscale=1):
    class_id = 0
    for count in np.nditer(vbox_count):
        if(count > 0 and count < 160):
            for i in range(count):
                box_id = vbox_ids[0, 0, i, class_id]
                xmin, xmax, ymin, ymax  = (
                    vboxes[0, 0, :, box_id] / (2**12))
                if labels[class_id] ==  "tennis racket":
                    box_start = (int(xmin * hscale), int(ymin *vscale))
                    box_end = (int(xmax *hscale), int(ymax*vscale))
                    txt_start = (int(xmin *hscale), int(ymin *vscale)-5)
                    color = COLORS[class_id]
                    cv2.rectangle(image, box_start,
                              box_end, color, 5)
                    image = cv2.putText(
                        image, labels[class_id], txt_start, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        class_id += 1
    cv2.imshow("frame", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit

    return


def tyolo_and_nms_run():
    # Pre-process input data or get a camera stream ready
    input_camera_stream = CameraStreamInput()

    # Inputs
    input_image_shape = (1, 1, 416, 1248)
    input_dtype = np.dtype(np.int8)

    # This is the output of the NN and input of NMS
    intermediate_shape = (1, 85, 1, 2535)
    intermediate_dtype = np.dtype(np.int32)

    # outputs
    nms_outputs = NmsOutputShapes()

    weights_host_tensor = np.fromfile(
        'tiny_yolo_epu_files/tiny_yolo_v3_epu.s.tensor0.bin', dtype=np.uint8)

    tyolo_kernel = quadric.get_kernel("./tiny_yolo_epu_files/tiny_yolo_v3_epu")
    nms_kernel = quadric.get_kernel("./nms/nms_epu")

    device_manager = quadric.DeviceManager()
    device = device_manager.get_device(
        quadric.SimulatorConfiguration(num_cores=16))
    weights_input_tensor = device.allocate_and_copy_ndarray(
        weights_host_tensor)

    # device space allocation
    # Allocate input tensor for NN
    input_device_tensor = device.allocate(input_image_shape, input_dtype)

    # Allocate output tensor of NN and input for NMS
    intermediate_device_tensor = device.allocate(
        intermediate_shape, intermediate_dtype)

    # Allocate output for NMS
    vboxes_device_tensor = device.allocate(
        nms_outputs.vboxes_shape, nms_outputs.vboxes_dtype)
    vbox_ids_device_tensor = device.allocate(
        nms_outputs.vbox_ids_shape, nms_outputs.vbox_ids_dtype)
    vbox_count_device_tensor = device.allocate(
        nms_outputs.vbox_count_shape, nms_outputs.vbox_count_dtype)
    score_tensor = device.allocate(nms_outputs.score_shape, nms_outputs.score_dtype)

    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    float_to_fp32 = lambda num, num_frac: int(num * (2 ** num_frac))
    
    nms_threshold = float_to_fp32(0.5, 12)
    score_threshold = float_to_fp32(0.5, 12)
    
    for orig_frame, frame_id in input_camera_stream:
        # Prepare the frame
        resized_orig_frame = cv2.resize(orig_frame, (416, 416))
        resized_reshaped_frame = resized_orig_frame.reshape(input_image_shape)
        # Copy image to device
        device.copy_ndarray_to_device(
            resized_reshaped_frame, input_device_tensor)
        device.load_kernel(tyolo_kernel)
        # Run YoloV3
        device.run_kernel("infer", (weights_input_tensor,
                          input_device_tensor, intermediate_device_tensor))
        # Run NMS
        device.load_kernel(nms_kernel)
        device.run_kernel("filter_boxes", (intermediate_device_tensor,
                                           vboxes_device_tensor, score_tensor, vbox_count_device_tensor, vbox_ids_device_tensor, score_threshold, nms_threshold))
        vbox_count = device.copy_ndarray_from_device(
            vbox_count_device_tensor, nms_outputs.vbox_count_shape, nms_outputs.vbox_count_dtype)
        vbox_ids = device.copy_ndarray_from_device(
            vbox_ids_device_tensor, nms_outputs.vbox_ids_shape, nms_outputs.vbox_ids_dtype)
        vboxes = device.copy_ndarray_from_device(
            vboxes_device_tensor, nms_outputs.vboxes_shape, nms_outputs.vboxes_dtype)

        draw_boxes(orig_frame, vboxes, vbox_count, vbox_ids, hscale=orig_frame.shape[1] / 416, vscale=orig_frame.shape[0] / 416)

if __name__ == "__main__":
    tyolo_and_nms_run()

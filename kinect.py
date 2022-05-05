import cv2
import numpy as np
from primesense import openni2


class KinectSensor:

    def __init__(self):
        openni2.initialize()
        self.device = openni2.Device.open_any()

        self.depth_stream = self.device.create_depth_stream()
        self.color_stream = self.device.create_color_stream()
        self.ir_stream = self.device.create_ir_stream()
        self.depth_stream.start()
        self.BRIGHTNESS_THRESHOLD = 7

    def get_color_image(self):
        self.color_stream.start()
        im_arr = self.color_stream.read_frame()
        raw_buf = im_arr.get_buffer_as_triplet()
        b_array = np.array([raw_buf[i][0] for i in range(640*480)])
        g_array = np.array([raw_buf[i][1] for i in range(640*480)])
        r_array = np.array([raw_buf[i][2]for i in range(640*480)])
        color_image = np.zeros([480, 640, 3])
        color_image[:, :, 0] = r_array.reshape(480, 640)
        color_image[:, :, 1] = g_array.reshape(480, 640)
        color_image[:, :, 2] = b_array.reshape(480, 640)
        color_image = color_image.astype(np.uint8)
        return np.fliplr(color_image)

    def get_ir_image(self):
        im_arr = self.ir_stream.read_frame()

        buf_array = np.array(im_arr.get_buffer_as_uint8())
        buf_array = buf_array.reshape((480, 640))

        ir_image = np.zeros([480, 640, 3])

        ir_image[:, :, 0] = buf_array
        ir_image[:, :, 1] = buf_array
        ir_image[:, :, 2] = buf_array

        return np.fliplr(ir_image)

    def get_image(self):
        color_img = self.get_color_image()
        if self.get_brightness(color_img) < self.BRIGHTNESS_THRESHOLD:
            self.color_stream.stop()
            self.ir_stream.start()
            ir_img = self.get_ir_image()
            self.ir_stream.stop()
            return ir_img
        else:
            return color_img

    def get_depth_map(self):
        im_arr = self.depth_stream.read_frame()
        raw_buf = im_arr.get_buffer_as_uint16()
        buf_array = np.array([raw_buf[i] for i in range(640*480)])

        depth_map = buf_array.reshape((480, 640))

        depth_map = depth_map * 0.1

        return np.fliplr(depth_map)

    def get_brightness(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = l * (10 / 255)
        y, x, z = frame.shape
        maxval = []
        count_percent = 9/100
        row_percent = int(count_percent * x)
        column_percent = int(count_percent * y)
        for i in range(1, x - 1):
            if i % row_percent == 0:
                for j in range(1, y - 1):
                    if j % column_percent == 0:
                        img_segment = l[i:i + 3, j:j + 3]
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                        maxval.append(maxVal)
        lenmaxVal = 0
        for i, val in enumerate(maxval):
            if val == 0:
                lenmaxVal += 1
        lenmaxVal = len(maxval) - lenmaxVal
        if lenmaxVal > 0:
            avg_maxval = round(sum(maxval) / lenmaxVal)
        else:
            avg_maxval = 0
        return avg_maxval

    def __del__(self):
        if getattr(self, "color_stream", None) is not None:
            self.color_stream.close()
        if getattr(self, "ir_stream", None) is not None:
            self.ir_stream.close()
        if getattr(self, "depth_stream", None) is not None:
            self.depth_stream.close()
        openni2.unload()

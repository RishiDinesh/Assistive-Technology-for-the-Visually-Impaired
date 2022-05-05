import cv2
import numpy as np

COCO_CLASSES = [['background', 0], ['person', 2], ['bicycle', 2], ['car', 2], ['motorcycle', 2], ['airplane', 0], ['bus', 2], ['train', 2],
                ['truck', 2], ['boat', 0], ['traffic light', 0], ['fire hydrant', 1], ['stop sign', 0], ['parking meter', 1], ['bench', 1],
                ['bird', 0], ['cat', 1], ['dog', 1], ['horse', 1], ['sheep', 1], ['cow', 1], ['elephant', 1], ['bear', 1], ['zebra', 1], ['giraffe', 1],
                ['backpack', 0], ['umbrella', 0], ['handbag', 0], ['tie', 0], ['suitcase', 0], ['frisbee', 0], ['skis', 0], ['snowboard', 0],
                ['sports ball', 0], ['kite', 0], ['baseball bat', 0], ['baseball glove', 0], ['skateboard', 1], ['surfboard', 1], ['tennis racket', 1],
                ['bottle', 0], ['wine glass', 0], ['cup', 0], ['fork', 0], ['knife', 0], ['spoon', 0], ['bowl', 0], ['banana', 0], ['apple', 0], ['sandwich', 0],
                ['orange', 0], ['broccoli', 0], ['carrot', 0], ['hot dog', 0], ['pizza', 0], ['donut', 0], ['cake', 0], ['chair', 1], ['couch', 1], ['potted plant', 1],
                ['bed', 1], ['dining table', 1], ['toilet', 1], ['tv', 1], ['laptop', 0], ['mouse', 0], ['remote', 0], ['keyboard', 0], ['cell phone', 0],
                ['microwave', 0], ['oven', 0], ['toaster', 0], ['sink', 0], ['refrigerator', 1], ['book', 0], ['clock', 0], ['vase', 1], ['scissors', 0],
                ['teddy bear', 0], ['hair drier', 0], ['toothbrush', 0]]

colors = [[56, 0, 255], [226, 255, 0], [0, 94, 255], [0, 37, 255], [0, 255, 94], [255, 226, 0], [0, 18, 255], [255, 151, 0], [170, 0, 255],
          [0, 255, 56], [255, 0, 75], [0, 75, 255], [0, 255, 169], [255, 0, 207], [75, 255, 0], [207, 0, 255], [37, 0, 255], [0, 207, 255],
          [94, 0, 255], [0, 255, 113], [255, 18, 0], [255, 0, 56], [18, 0, 255], [0, 255, 226], [170, 255, 0], [255, 0, 245], [151, 255, 0],
          [132, 255, 0], [75, 0, 255], [151, 0, 255], [0, 151, 255], [132, 0, 255], [0, 255, 245], [255, 132, 0], [226, 0, 255], [255, 37, 0],
          [207, 255, 0], [0, 255, 207], [94, 255, 0], [0, 226, 255], [56, 255, 0], [255, 94, 0], [255, 113, 0], [0, 132, 255], [255, 0, 132],
          [255, 170, 0], [255, 0, 188], [113, 255, 0], [245, 0, 255], [113, 0, 255], [255, 188, 0], [0, 113, 255], [255, 0, 100], [0, 56, 255],
          [255, 0, 113], [0, 255, 188], [255, 0, 94], [255, 0, 18], [18, 255, 0], [0, 255, 132], [0, 188, 255], [0, 245, 255], [0, 169, 255],
          [37, 255, 0], [255, 0, 151], [188, 0, 255], [0, 255, 37], [0, 255, 0], [255, 0, 170], [255, 0, 37], [255, 75, 0], [10, 0, 255],
          [255, 207, 0], [255, 0, 226], [255, 245, 0], [188, 255, 0], [0, 255, 18], [0, 255, 75], [0, 255, 151], [255, 56, 0], [245, 255, 0]]


class Yolact():
    def __init__(self, confThreshold=0.5, nmsThreshold=0.5, keep_top_k=200):
        self.target_size = 550
        self.MEANS = np.array([103.94, 116.78, 123.68],
                              dtype=np.float32).reshape(1, 1, 3)
        self.STD = np.array([57.38, 57.12, 58.40],
                            dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNet('convert-onnx\yolact_base_54_800000.onnx')
        self.confidence_threshold = confThreshold
        self.nms_threshold = nmsThreshold
        self.keep_top_k = keep_top_k
        self.conv_ws = [69, 35, 18, 9, 5]
        self.conv_hs = [69, 35, 18, 9, 5]
        self.aspect_ratios = [1, 0.5, 2]
        self.scales = [24, 48, 96, 192, 384]
        self.variances = [0.1, 0.2]
        self.last_img_size = None
        self.priors = self.make_priors()

    def make_priors(self):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        if self.last_img_size != (self.target_size, self.target_size):
            prior_data = []
            for conv_w, conv_h, scale in zip(self.conv_ws, self.conv_hs, self.scales):
                for i in range(conv_h):
                    for j in range(conv_w):
                        # +0.5 because priors are in center-size notation
                        cx = (j + 0.5) / conv_w
                        cy = (i + 0.5) / conv_h
                        for ar in self.aspect_ratios:
                            ar = np.sqrt(ar)
                            w = scale * ar / self.target_size
                            h = scale / ar / self.target_size
                            h = w
                            prior_data += [cx, cy, w, h]
            self.priors = np.array(prior_data).reshape(-1, 4)
            self.last_img_size = (self.target_size, self.target_size)
        return self.priors

    def decode(self, loc, priors, img_w, img_h):
        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * self.variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * self.variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]

        # crop
        np.where(boxes[:, 0] < 0, 0, boxes[:, 0])
        np.where(boxes[:, 1] < 0, 0, boxes[:, 1])
        np.where(boxes[:, 2] > 1, 1, boxes[:, 2])
        np.where(boxes[:, 3] > 1, 1, boxes[:, 3])

        # decode to img size
        boxes[:, 0] *= img_w
        boxes[:, 1] *= img_h
        boxes[:, 2] = boxes[:, 2] * img_w + 1
        boxes[:, 3] = boxes[:, 3] * img_h + 1
        return boxes

    def detect(self, srcimg, depthimg):
        result = []
        img_h, img_w = srcimg.shape[:2]
        img = cv2.resize(srcimg, (self.target_size, self.target_size),interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img = (img - self.MEANS) / self.STD
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        # Sets the input to the network
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        loc_data, conf_preds, mask_data, proto_data = self.net.forward(self.net.getUnconnectedOutLayersNames())
        cur_scores = conf_preds[:, 1:]
        classid = np.argmax(cur_scores, axis=1)
        # conf_scores = np.max(cur_scores, axis=1)
        conf_scores = cur_scores[range(cur_scores.shape[0]), classid]
        # filter by confidence_threshold
        keep = conf_scores > self.confidence_threshold
        conf_scores = conf_scores[keep]
        classid = classid[keep]
        loc_data = loc_data[keep, :]
        prior_data = self.priors[keep, :]
        masks = mask_data[keep, :]
        boxes = self.decode(loc_data, prior_data, img_w, img_h)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf_scores.tolist(
        ), self.confidence_threshold, self.nms_threshold, top_k=self.keep_top_k)
        indices = indices.flatten() if len(indices) != 0 else []
        for idx in indices:
            bbox = boxes[idx, :].astype(np.int32).tolist()
            # generate mask
            mask = proto_data @ masks[idx, :].reshape(-1, 1)
            mask = 1 / (1 + np.exp(-mask))  # sigmoid
            # Scale masks up to the full image
            mask = cv2.resize(mask.squeeze(), (img_w, img_h),interpolation=cv2.INTER_LINEAR)
            mask = mask > 0.5
            depth = sum(depthimg[mask])/len(depthimg[mask])
            class_info = COCO_CLASSES[classid[idx]+1]
            result.append([bbox, class_info[0], class_info[1], depth])
        return result

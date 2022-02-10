#

import numpy as np
import cv2
import torch

try:
    import toolkit.customsnippets as customsnippets
except:
    import customsnippets
try:
    import toolkit.methodspaths as methodspaths
except:
    import methodspaths


def preprocess_img(in_img):
    hardcodedsize = (640, 640)
    probe_img = cv2.resize(in_img, hardcodedsize, cv2.INTER_AREA)
    probe_img = np.moveaxis(probe_img, [0, 1, 2], [2, 1, 0])
    probe_img = np.moveaxis(probe_img, [0, 1, 2], [0, 2, 1])
    probe_img = np.array([probe_img, ]).astype(np.float32)
    return probe_img


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////
class PoseDetection:
    def __init__(self, method='yolox', architecture='pytorch'):
        self._methods_pool = ['yolox']
        self._architecture_pool = ['tflite', 'pytorch', 'onnx']
        self._cheatsheet = {'tflite': 0,
                            'pytorch': 1,
                            'onnx': 2}
        self._method = method
        self._architecture = architecture
        self._setup()

    def _setup(self):
        mthd_ok = any(self._method == word for word in self._methods_pool)
        rchtctr_ok = any(self._architecture == word for word in self._architecture_pool)

        if mthd_ok and rchtctr_ok:
            if self._cheatsheet[self._architecture] == 0:
                self.detector = YoloxDetectionTFLITE()
            elif self._cheatsheet[self._architecture] == 1:
                self.detector = YoloxDetectionPytorch()
            elif self._cheatsheet[self._architecture] == 2:
                self.detector = YoloxDetectionONNX()
            else:
                raise TypeError('not yet')
        else:
            raise TypeError('method or architecture not valid')

    def detect(self, in_img):
        return self.detector.detect(in_img)


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# ----------------------------------------------------------------------------------------------
class YoloxDetectionPytorch:
    def __init__(self):
        _paths = methodspaths.methodsDict['yolox_Paths']
        self.config = _paths.cfg
        self.checkpoint = _paths.pth
        self.decodeBB = customsnippets.GetPoseDetectionBBNN()
        self._setup()

    def _setup(self):
        from mmdet.apis import init_detector
        self.model = customsnippets.build_custom_yolox_from_model(
            init_detector(self.config, self.checkpoint, device='cpu'))
        self.decodeBB.eval()

    def detect(self, in_img):
        probe_img = preprocess_img(in_img)
        in_tensor = torch.from_numpy(probe_img).to('cpu')
        with torch.no_grad():
            out = self.model(in_tensor)

        return out.detach().numpy()


# ----------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
class YoloxDetectionONNX:
    def __init__(self):
        _paths = methodspaths.methodsDict['yolox_Paths']
        self.onnx_file = _paths.onnx
        self._setup()

    def _setup(self):
        import onnxruntime
        self.model = onnxruntime.InferenceSession(self.onnx_file)

    def detect(self, in_img):
        probe_img = preprocess_img(in_img)
        c_input = {self.model.get_inputs()[0].name: probe_img}
        c_results = self.model.run(None, c_input)

        return c_results[0]


# ----------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
class YoloxDetectionTFLITE:
    def __init__(self):
        _paths = methodspaths.methodsDict['yolox_Paths']
        self.tflite_file = _paths.tflite
        self._setup()

    def _setup(self):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(self.tflite_file)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self._input_details = self.interpreter.get_input_details()
        self._input_shape = self._input_details[0]['shape']
        self._output_details = self.interpreter.get_output_details()

    def detect(self, in_img):
        probe_img = preprocess_img(in_img)

        self.interpreter.set_tensor(self._input_details[0]['index'], probe_img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self._output_details[0]['index'])

        return output


# ----------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------
def test():
    pclass = PoseDetection(method='yolox', architecture='vpu')
    wally = 'here'


# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #test()
    pass


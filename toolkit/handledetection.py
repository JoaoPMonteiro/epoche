#


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////
class PoseDetection:
    def __init__(self, method='yolox', architecture='pytorch'):
        self._methods_pool = ['yolox']
        self._architecture_pool = ['tflite', 'pytorch', 'onnx', 'vpu']
        self._method = method
        self._architecture = architecture
        self._setup()


    def _setup(self):
        mthd_ok = any(self._method == word for word in self._methods_pool)
        rchtctr_ok = any(self._architecture == word for word in self._architecture_pool)

        if mthd_ok and rchtctr_ok:
            wally = 'here'
        else:
            raise TypeError('method or architecture not valid')

        return 0


    def detect(self, in_img):
        return 0
# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# ----------------------------------------------------------------------------------------------
class YoloxDetectionPytorch:
    def __init__(self):
        self.config = '../methods/YOLOX/config_files/yolox_s_8x8_300e_coco.py'
        self.checkpoint = '../methods/YOLOX/pth_files/yolox_tiny_8x8_300e_coco.pth'
        self._setup()


    def _setup(self):
        # do setup
        return 0


    def detect(self, in_img):
        return 0
# ----------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
class YoloxDetectionONNX:
    def __init__(self):
        self._setup()


    def _setup(self):
        # do setup
        return 0


    def detect(self, in_img):
        return 0
# ----------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
class YoloxDetectionVPU:
    def __init__(self):
        self._setup()


    def _setup(self):
        # do setup
        return 0


    def detect(self, in_img):
        return 0
# ----------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------
def test():
    pclass = PoseDetection(method='yolox', architecture='vpu')
    wally = 'here'


# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    test()

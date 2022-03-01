import depthai as dai


class PoseEdgeWorker:
    def __init__(self, method, detector):
        self._method = method
        self._detector = detector
        self._det_pool = ['yolox']
        self._cheatsheet = {'blazepose_lite': 0,
                            'blazepose_full': 0,
                            'blazepose_heavy': 0,
                            'hrnet_vpose': 1,
                            'hrnet_sbl': 1,
                            'hrnet_gcn': 1,
                            'hrnet_stgcn': 1,
                            'hrnet_poseaugvpose': 1,
                            'hrnet_poseaugsbl': 1,
                            'hrnet_poseauggcn': 1,
                            'hrnet_poseaugstgcn': 1}
        self._setup()

    def _setup(self):
        dt_ok = any(self._detector == word for word in self._det_pool)
        if dt_ok:
            try:
                _selector = self._cheatsheet[self._method]
                if _selector == 0:
                    self._est = PoseEdgeWorkerSingleStage(method=self._method,
                                                          detector=self._detector)
                elif _selector == 1:
                    x_mthd = self._method.split('_')
                    self._est = PoseEdgeWorkerTwoSteps(first_stage=x_mthd[0],
                                                       second_stage=x_mthd[1],
                                                       detector=self._detector)
            except:
                raise TypeError('selection not valid')
            else:
                raise TypeError('selection not valid')
        else:
            raise TypeError('method or architecture not valid')

    def estimate(self, in_img):
        return 0


class PoseEdgeWorkerSingleStage:
    def __init__(self, method, detector):
        self._method = method
        self._detector = detector
        self._setup()

    def _setup(self):
        self.path_to_detector = ''
        self.path_to_estimator = ''

    def estimate(self, in_img):
        return 0


class PoseEdgeWorkerTwoSteps:
    def __init__(self, first_stage, second_stage, detector):
        self._first_stage = first_stage
        self._second_stage = second_stage
        self._detector = detector
        self._setup()

    def _setup(self):
        self.path_to_detector = ''
        self.path_to_estimator = ''
        self.path_to_lifter = ''
        self._input_shape_detection = (640, 640)  # yolox, todo: move all constants to somewhere else
        self._input_shape_estimation = (256, 256)
        self._setup_pipeline()

    def _setup_pipeline(self):
        nr_inference_threads = 1
        blocking_opt = True
        nr_per_threads = 1
        # ----------------------------------------------------------------

        l_pipeline = dai.Pipeline()
        l_pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

        # setting up camera node
        camera_rgb = l_pipeline.create(dai.node.ColorCamera)
        camera_rgb.setPreviewSize(self._input_shape_detection[0], self._input_shape_detection[1])
        camera_rgb.setInterleaved(False)
        camera_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # setting up person detector stage
        detector_nn = l_pipeline.create(dai.node.NeuralNetwork)
        detector_nn.setBlobPath(self.path_to_detector)
        detector_nn.setNumInferenceThreads(nr_inference_threads)
        detector_nn.input.setBlocking(blocking_opt)
        detector_nn.setNumNCEPerInferenceThread(nr_per_threads)

        # setting up preprocess for estimation part I/II (crop) and II/II (resize)
        pre_0_manip = l_pipeline.create(dai.node.ImageManip)
        pre_0_manip.initialConfig.setCropRect(0., 0., 1., 1.)
        pre_0_manip.initialConfig.setResize(self._input_shape_estimation[0], self._input_shape_estimation[0])
        pre_0_manip.setMaxOutputFrameSize(self._input_shape_estimation[0]*self._input_shape_estimation[1]*3)
        pre_0_manip.inputImage.setQueueSize(1)
        pre_0_manip.inputImage.setBlocking(blocking_opt)
        pre_0_manip.setWaitForConfigInput(False)

    def estimate(self, in_img):
        return 0

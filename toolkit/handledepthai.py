import depthai as dai
import numpy as np
import cv2
try:
    import toolkit.methodspaths as methodspaths
except:
    import methodspaths


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
            raise TypeError('method or architecture not valid')

    def estimate(self, in_img):
        _out = self._est.estimate(in_img)
        return _out


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
        zero_stage_paths = self._detector + '_Paths'
        self._paths0 = methodspaths.methodsDict[zero_stage_paths]
        first_stage_paths = self._first_stage + '_Paths'
        self._paths1 = methodspaths.methodsDict[first_stage_paths]
        second_stage_paths = self._second_stage + '_Paths'
        self._paths2 = methodspaths.methodsDict[second_stage_paths]

        self.path_to_detector = self._paths0.blob
        self.path_to_estimator = self._paths1.blob
        self.path_to_lifter = self._paths2.blob
        self._input_shape_detection = (640, 640)  # yolox, todo: move all constants to somewhere else
        self._input_shape_estimation = (256, 256)
        self._pipeline = self._setup_pipeline()

    def _setup_pipeline(self):
        nr_inference_threads = 1
        blocking_opt = True
        nr_per_threads = 1

        # ----------------------------------------------------------------

        l_pipeline = dai.Pipeline()
        l_pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

        # setting up camera node
        cam_option = False
        if cam_option:
            camera_rgb = l_pipeline.create(dai.node.ColorCamera)
            camera_rgb.setPreviewSize(self._input_shape_detection[0], self._input_shape_detection[1])
            camera_rgb.setInterleaved(False)
            camera_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        else:
            xRGBIn = l_pipeline.create(dai.node.XLinkIn)
            xRGBIn.setStreamName("input_img")

        # setting up person detector stage
        detector_nn = l_pipeline.create(dai.node.NeuralNetwork)
        detector_nn.setBlobPath(self.path_to_detector)
        detector_nn.setNumInferenceThreads(nr_inference_threads)
        detector_nn.input.setBlocking(blocking_opt)
        detector_nn.setNumNCEPerInferenceThread(nr_per_threads)

        if False:
            # setting up preprocess for estimation part I/II (crop) and II/II (resize)
            pre_0_manip = l_pipeline.create(dai.node.ImageManip)
            pre_0_manip.initialConfig.setCropRect(0., 0., 1., 1.)
            pre_0_manip.initialConfig.setResize(self._input_shape_estimation[0], self._input_shape_estimation[0])
            pre_0_manip.setMaxOutputFrameSize(self._input_shape_estimation[0]*self._input_shape_estimation[1]*3)
            pre_0_manip.inputImage.setQueueSize(1)
            pre_0_manip.inputImage.setBlocking(blocking_opt)
            pre_0_manip.setWaitForConfigInput(False)

            # setting up 2d pose landmark estimator stage
            estimator_nn = l_pipeline.create(dai.node.NeuralNetwork)
            estimator_nn.setBlobPath(self.path_to_estimator)
            estimator_nn.setNumInferenceThreads(nr_inference_threads)
            estimator_nn.input.setBlocking(blocking_opt)
            estimator_nn.setNumNCEPerInferenceThread(nr_per_threads)

            # setting up 3d pose lifter stage
            lifter_nn = l_pipeline.create(dai.node.NeuralNetwork)
            lifter_nn.setBlobPath(self.path_to_lifter)
            lifter_nn.setNumInferenceThreads(nr_inference_threads)
            lifter_nn.input.setBlocking(blocking_opt)
            lifter_nn.setNumNCEPerInferenceThread(nr_per_threads)

        xout_bb = l_pipeline.create(dai.node.XLinkOut)
        xout_bb.setStreamName("bb_out")

        # linking stuff
        if cam_option:
            camera_rgb.preview.link(detector_nn.input)
        else:
            xRGBIn.out.link(detector_nn.input)
        detector_nn.out.link(xout_bb.input)

        return l_pipeline

    def estimate(self, in_img):
        probe_img_0 = cv2.resize(in_img, (self._input_shape_detection[0], self._input_shape_detection[1]),
                                 interpolation=cv2.INTER_AREA)
        input_i = np.moveaxis(probe_img_0, [0, 1, 2], [2, 1, 0])
        input_i = np.moveaxis(input_i, [0, 1, 2], [0, 2, 1])
        input_i = np.array([input_i, ]).astype(np.uint8)
        with dai.Device(self._pipeline) as device:

            device.setLogLevel(dai.LogLevel.WARN)
            device.setLogOutputLevel(dai.LogLevel.WARN)
            q_bb_out = device.getOutputQueue("bb_out", maxSize=1, blocking=False)
            q_img_in = device.getInputQueue("input_img")
            try:
                img_nn = dai.NNData()
                #planar_nn = input_i.astype(np.float64).flatten()
                planar_nn = input_i.flatten()
                img_nn.setLayer("input_1", planar_nn)
                q_img_in.send(img_nn)

                bb_outq = q_bb_out.get()
                if bb_outq is not None:
                    bb_ou3456yu = bb_outq.getAllLayers()
                    #bb_ou = bb_outq.getLayerFp16('902')
                    bb_ou = bb_outq.getLayerFp16('1049')
                    wally = 'here'
            except:
                raise TypeError('something')
        return 0

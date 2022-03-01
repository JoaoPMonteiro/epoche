#
import torch
import importlib
import sys
import numpy as np
import cv2
import onnxruntime

sys.path.append('./')

# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////
try:
    import toolkit.customsnippets as customsnippets
except:
    import customsnippets
try:
    import toolkit.methodspaths as methodspaths
except:
    import methodspaths
try:
    import toolkit.handledetection as handledetection
except:
    import handledetection
try:
    import toolkit.handledepthai as handledepthai
except:
    import handledepthai


class PosePipeline:
    def __init__(self, method, architecture, detector):
        self._methods_pool = ['blazepose_lite',
                              'blazepose_full',
                              'blazepose_heavy',
                              'hrnet_vpose',
                              'hrnet_sbl',
                              'hrnet_gcn',
                              'hrnet_stgcn',
                              'hrnet_poseaugvpose',
                              'hrnet_poseaugsbl',
                              'hrnet_poseauggcn',
                              'hrnet_poseaugstgcn']
        self._architecture_pool = ['tflite', 'pytorch', 'onnx', 'vpu']
        self._det_pool = ['none', 'yolox']
        self._method = method
        self._architecture = architecture
        self._detector = detector
        self._setup()

    def _setup(self):
        mthd_ok = any(self._method == word for word in self._methods_pool)
        rchtctr_ok = any(self._architecture == word for word in self._architecture_pool)
        dt_ok = any(self._detector == word for word in self._det_pool)

        if mthd_ok and rchtctr_ok and dt_ok:
            if self._architecture == 'vpu':
                self.pipe_vpu = handledepthai.PoseEdgeWorker(method=self._method, detector=self._detector)
            else:
                self.detectorInst = handledetection.PoseDetection(method=self._detector,
                                                                  architecture=self._architecture)
                self.estimatorInst = PoseEstimation(method=self._method,
                                                    architecture=self._architecture)
        else:
            raise TypeError('method or architecture not valid')

    def estimate(self, in_img):
        if self._architecture == 'vpu':
            outd_pose = self.pipe_vpu.estimate(in_img)
        else:
            outbb = self.detectorInst.detect(in_img)
            outd_pose = self.estimatorInst.estimate(in_img, outbb)
            outd_pose = outd_pose - outd_pose[0, :]

        return outd_pose


class PoseEstimation:
    def __init__(self, method, architecture):
        self._methods_pool = ['blazepose_lite',
                              'blazepose_full',
                              'blazepose_heavy',
                              'hrnet_vpose',
                              'hrnet_sbl',
                              'hrnet_gcn',
                              'hrnet_stgcn',
                              'hrnet_poseaugvpose',
                              'hrnet_poseaugsbl',
                              'hrnet_poseauggcn',
                              'hrnet_poseaugstgcn']
        self._architecture_pool = ['tflite', 'pytorch', 'onnx']
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
        self._method = method
        self._architecture = architecture
        self._setup()

    def _setup(self):
        mthd_ok = any(self._method == word for word in self._methods_pool)
        rchtctr_ok = any(self._architecture == word for word in self._architecture_pool)

        if mthd_ok and rchtctr_ok:
            _selector = self._cheatsheet[self._method]
            if _selector == 0:
                self._est = SingleStage3D(method=self._method, architecture=self._architecture)
            elif _selector == 1:
                x_mthd = self._method.split('_')
                self._est = TwoStage3D(first_stage=x_mthd[0], second_stage=x_mthd[1], architecture=self._architecture)
            else:
                raise TypeError('selection not valid')
        else:
            raise TypeError('method or architecture not valid')

        return 0

    def estimate(self, in_img, in_bb):
        return self._est.estimate(in_img, in_bb)


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////
class PoseEstimation2DPY:
    def __init__(self):
        self._model = customsnippets.build_custom_hrnet()

    def estimate(self, in_img):
        in_tensor = torch.from_numpy(in_img)
        _out = self._model(in_tensor)
        return _out.detach().numpy()


class OnnxWorker:
    def __init__(self, onnx_file):
        self._model = onnxruntime.InferenceSession(onnx_file)

    def estimate(self, in_img):
        probe_img = in_img
        c_input = {self._model.get_inputs()[0].name: probe_img}
        c_results = self._model.run(None, c_input)
        return c_results[0]


class PoseEstimation2D:
    def __init__(self, method, architecture):
        self._methods_pool = ['hrnet']
        self._architecture_pool = ['pytorch', 'onnx']
        self._method = method
        self._architecture = architecture
        self._setup()

    def _setup(self):
        mthd_ok = any(self._method == word for word in self._methods_pool)
        rchtctr_ok = any(self._architecture == word for word in self._architecture_pool)

        if mthd_ok and rchtctr_ok:
            _id_paths = self._method + '_Paths'
            self._paths = methodspaths.methodsDict[_id_paths]
            id_arch = [jj for jj, ii in enumerate(self._architecture_pool) if ii == self._architecture]
            if id_arch[0] == 0:
                self._model = PoseEstimation2DPY()
            elif id_arch[0] == 1:
                self._model = OnnxWorker(self._paths.onnx)
            else:
                raise TypeError('selection not valid')
        else:
            raise TypeError('method or architecture not valid')

    def estimate(self, in_img):
        out_landmarks = self._model.estimate(in_img)
        # def drawPreds(image_in, preds_in):
        #    image_out = image_in.copy()
        #    image_out = np.moveaxis(image_out, [0, 1, 2, 3], [0, 3, 2, 1])
        #    image_out = np.moveaxis(image_out, [0, 1, 2, 3], [0, 2, 1, 3])
        #    image_out2 = image_out[0].astype(np.uint8).copy()
        #    probe_test = np.zeros((640, 640, 3), np.uint8)
        #    for set in preds_in:
        #        for point in set:
        #            x = int(point[0])
        #            y = int(point[1])
        #            image_out = cv2.circle(image_out2, (x, y), 3, (255, 0, 0), thickness=-1)
        #            #(img, center, radius, color, thickness=None, lineType=None, shift=None):
        #    # cv2.imwrite('onetime.png', image_out)

        #    return image_out

        # asd = drawPreds(in_img, out_landmarks)
        return out_landmarks


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////
class PoseLifting:
    def __init__(self, method, architecture):
        self._methods_pool = ['vpose',
                              'sbl',
                              'gcn',
                              'stgcn',
                              'poseaugvpose',
                              'poseaugsbl',
                              'poseauggcn',
                              'poseaugstgcn']
        self._architecture_pool = ['pytorch', 'onnx']
        self._cheatsheet = {'vpose': 0,
                            'sbl': 0,
                            'gcn': 0,
                            'stgcn': 0,
                            'poseaugvpose': 1,
                            'poseaugsbl': 1,
                            'poseauggcn': 1,
                            'poseaugstgcn': 1}
        self._method = method
        self._architecture = architecture
        self._paths = methodspaths.MethodsPaths()
        self._model = None
        self._setup()

    def _setup(self):
        mthd_ok = any(self._method == word for word in self._methods_pool)
        rchtctr_ok = any(self._architecture == word for word in self._architecture_pool)

        if mthd_ok and rchtctr_ok:
            _id_paths = self._method + '_Paths'
            _probe = self._cheatsheet[self._method]
            if _probe == 1:
                id_method = self._method[7:]
            else:
                id_method = self._method
            self._paths = methodspaths.methodsDict[_id_paths]
            id_arch = [jj for jj, ii in enumerate(self._architecture_pool) if ii == self._architecture]
            self._id_method = id_method
            if id_arch[0] == 0:
                self._init_pytorch(id_method)
            elif id_arch[0] == 1:
                self._init_onnx()
            else:
                raise TypeError('selection not valid')
        else:
            raise TypeError('method or architecture not valid')

        return 0

    def _init_pytorch(self, selected_method):
        if selected_method == 'sbl':
            self._sbl()
        elif selected_method == 'gcn':
            self._gcn()
        elif selected_method == 'stgcn':
            self._stgcn()
        elif selected_method == 'vpose':
            self._vpose()
        else:
            raise TypeError('selection not valid')
        self._model.eval()

    def _sbl(self):
        l_paths = self._paths
        evaluate = l_paths.pth
        x2 = l_paths.cfg
        x2 = x2.replace('../', '')
        x2 = x2.replace('/', '.')
        x2 = x2.replace('.py', '')

        LinearModel = getattr(importlib.import_module(x2), 'LinearModel')
        init_weights = getattr(importlib.import_module(x2), 'init_weights')

        stages = 2
        num_joints = 16
        dropout = 0.25
        model_1 = LinearModel(num_joints * 2, (num_joints - 1) * 3, num_stage=stages, p_dropout=dropout)
        model_1.apply(init_weights)
        ckpt = torch.load(evaluate)

        try:
            model_1.load_state_dict(ckpt['state_dict'])
        except:
            model_1.load_state_dict(ckpt['model_pos'])

        self._model = model_1

    def _vpose(self):
        l_paths = self._paths
        evaluate = l_paths.pth

        x = l_paths.cfg
        x = x.replace('../', '')
        x = x.replace('/', '.')
        x = x.replace('.py', '')

        TemporalModelOptimized1f = getattr(importlib.import_module(x), 'TemporalModelOptimized1f')
        stages = 4
        filter_widths = [1]
        for stage_id in range(stages):
            filter_widths.append(1)
        # -----------------
        model_1 = TemporalModelOptimized1f(16, 2, 15, filter_widths=filter_widths, causal=False, dropout=0.25,
                                           channels=1024)

        x2 = methodspaths.methodsDict['sbl_Paths'].cfg
        x2 = x2.replace('../', '')
        x2 = x2.replace('/', '.')
        x2 = x2.replace('.py', '')

        init_weights = getattr(importlib.import_module(x2), 'init_weights')
        model_1.apply(init_weights)
        ckpt = torch.load(evaluate)

        try:
            model_1.load_state_dict(ckpt['state_dict'])
        except:
            model_1.load_state_dict(ckpt['model_pos'])

        self._model = model_1

    def _gcn(self):

        l_paths = self._paths
        evaluate = l_paths.pth

        x2 = methodspaths.methodsDict['sbl_Paths'].cfg
        x2 = x2.replace('../', '')
        x2 = x2.replace('/', '.')
        x2 = x2.replace('.py', '')

        init_weights = getattr(importlib.import_module(x2), 'init_weights')

        stages = 4
        dropout = 0.25
        adj_t = torch.tensor([[0.2500, 0.2500, 0.0000, 0.0000, 0.2500, 0.0000, 0.0000, 0.2500, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.3333, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.3333, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.5000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2000, 0.2000,
                               0.2000, 0.2000, 0.0000, 0.0000, 0.2000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000,
                               0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333,
                               0.0000, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.3333],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.5000]])

        model_1 = customsnippets.CustomSemGCN(adj_t, 128, num_layers=stages, p_dropout=dropout, nodes_group=None)

        model_1.apply(init_weights)
        ckpt = torch.load(evaluate)

        try:
            model_1.load_state_dict(ckpt['state_dict'])
        except:
            model_1.load_state_dict(ckpt['model_pos'])

        self._model = model_1

    def _stgcn(self):
        l_paths = self._paths
        evaluate = l_paths.pth

        x2 = methodspaths.methodsDict['sbl_Paths'].cfg
        x2 = x2.replace('../', '')
        x2 = x2.replace('/', '.')
        x2 = x2.replace('.py', '')

        init_weights = getattr(importlib.import_module(x2), 'init_weights')

        dropout = 0.25

        model_1 = customsnippets.CustomWrapSTGCN(p_dropout=dropout)

        model_1.apply(init_weights)
        ckpt = torch.load(evaluate)

        try:
            model_1.load_state_dict(ckpt['state_dict'])
        except:
            model_1.load_state_dict(ckpt['model_pos'])

        self._model = model_1

    def _init_onnx(self):
        l_paths = self._paths
        pfof = l_paths.onnx
        self._model = OnnxWorker(pfof)
        #wally = 'here'

    def estimate(self, in_img):
        #if self._id_method == 'gcn':
        #    asdasdasd = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 13, 14, 15, 10, 11, 12]
        #    #asdasdasd = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        #    in_img = in_img[asdasdasd]  # wally
        #import toolkit.assortedroutines
        #toolkit.assortedroutines.another_2d_plot2(in_img)
        #wally = 'here'

        if self._architecture == 'pytorch':
            l_in_1 = torch.from_numpy(in_img)  # temp
            l_in_2 = l_in_1.view(1, -1).float()
            with torch.no_grad():
                oustp = self._model(l_in_2)
            outpn = oustp.detach().numpy()
        elif self._architecture == 'onnx':
            p_img = np.array([in_img, ], dtype=np.float32)
            out_0 = self._model.estimate(p_img)
            outpn = out_0
        else:
            raise TypeError('selection not valid')

        #if self._id_method == 'gcn':
        #    asdasdasd = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 13, 14, 15, 10, 11, 12]
        #    #asdasdasd = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        #    outpn[0] = outpn[0][asdasdasd]  # wally

        return outpn

    def get_model(self):
        return self._model


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////

def from_blazepose_to_16(in_body):
    arr = np.full((17, 3), 0).astype(np.float32)

    indexes = [[24, 23],  # 0  - Hip
               24,  # 1  - RHip
               26,  # 2  - RKnee
               [28, 30, 32],  # 3  - RFoot /32
               23,  # 4  - LHip
               25,  # 5  - LKnee
               [27, 29, 31],  # 6  - LFoot /31
               [11, 12, 23, 24],  # 7  - Spine
               [11, 12],  # 8  - Thorax
               0,  # 9  - Neck/Nose
               [9, 10, 7, 8, 3, 6, 2, 5, 1, 4, 0],  # 10 - Head [1, 4]
               11,  # 11 - LShoulder
               13,  # 12 - LElbow
               [15, 17, 19, 21],  # 13 - LWrist
               12,  # 14 - RShoulder
               14,  # 15 - RElbow
               [16, 18, 20, 22]]  # 16 - RWrist

    for count, value in enumerate(indexes):
        if type(value) is list:
            xyz_avalues = [in_body[probe, :] for probe in value]
            average = [sum(x) / len(x) for x in zip(*xyz_avalues)]
            # a = 1
            # do average
            # av2 = np.average(np.array(xyz_avalues), 0)
            arr[count, :] = average
        else:
            arr[count, :] = in_body[value]

    # remove nose
    rem_nose = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
    no_nose = arr[rem_nose]
    # outout = no_nose - no_nose[0, :]

    return no_nose


# //////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------------------------------------------------------------------------
class SStfl:
    '''
    single stage tensor flow lite estimator
    '''

    def __init__(self, in_method):
        self._paths = methodspaths.methodsDict
        self._method = in_method
        self.setup()

    def setup(self):
        import tensorflow as tf
        tflite_file = self._paths[self._method + '_Paths'].tflite
        self.interpreter = tf.lite.Interpreter(tflite_file)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self._input_details = self.interpreter.get_input_details()
        self._input_shape = self._input_details[0]['shape']
        self._output_details = self.interpreter.get_output_details()

    def estimate(self, in_img):
        self.interpreter.set_tensor(self._input_details[0]['index'], in_img)
        self.interpreter.invoke()
        # outputid = self.interpreter.get_tensor(self._output_details[0]['index'])
        output3d = self.interpreter.get_tensor(self._output_details[4]['index'])
        output3d_out = output3d.reshape(39, 3)
        post_3d = from_blazepose_to_16(output3d_out)
        # import toolkit.assortedroutines
        # toolkit.assortedroutines.another_3d_plot(post_3d)
        # toolkit.assortedroutines.another_3d_plot(output3d_out)
        post_3d = post_3d - post_3d[0, :]
        return post_3d


class SSo:
    '''
    single stage onnx estimator
    '''

    def __init__(self, in_method):
        self._paths = methodspaths.methodsDict
        self._method = in_method
        self.setup()

    def setup(self):
        import onnxruntime
        onnx_file = self._paths[self._method + '_Paths'].onnx
        self.model = onnxruntime.InferenceSession(onnx_file)

    def estimate(self, in_img):
        c_input = {self.model.get_inputs()[0].name: in_img}
        c_results = self.model.run(None, c_input)
        output3d = c_results[4]
        output3d_out = output3d.reshape(39, 3)
        post_3d = from_blazepose_to_16(output3d_out)
        post_3d = post_3d - post_3d[0, :]
        return post_3d


class SingleStage3D:
    def __init__(self, method, architecture):
        self._methods_pool = ['blazepose_lite',
                              'blazepose_full',
                              'blazepose_heavy']
        self._architecture_pool = ['tflite', 'onnx']
        self._method = method
        self._architecture = architecture
        self.input_size = [1, 256, 256, 3]
        self._setup()

    def _init_proc(self):
        if self._architecture == 'tflite':
            self.worker = SStfl(self._method)
        elif self._architecture == 'onnx':
            self.worker = SSo(self._method)
        else:
            raise TypeError('architecture not valid')

    def _setup(self):
        mthd_ok = any(self._method == word for word in self._methods_pool)
        rchtctr_ok = any(self._architecture == word for word in self._architecture_pool)

        if mthd_ok and rchtctr_ok:
            self._init_proc()
        else:
            raise TypeError('method or architecture not valid')

        return 0

    def handleroi(self, in_img, in_bb):
        imginfo = in_img.shape
        iw = imginfo[0]
        ih = imginfo[1]
        # ms_bbox_result = in_bb
        # persons = ms_bbox_result[0]
        person_bb = in_bb[:4]
        person_bb = person_bb.astype(int)
        xis_factor_x = 0.05
        xis_factor_y = 0.05
        mm_y_min = int(person_bb[1] * (1 - xis_factor_y))
        if mm_y_min < 0:
            mm_y_min = 0
        mm_y_max = int(person_bb[3] * (1 + xis_factor_y))
        if mm_y_max > ih:
            mm_y_max = int(ih)
        mm_x_min = int(person_bb[0] * (1 - xis_factor_x))
        if mm_x_min < 0:
            mm_x_min = 0
        mm_x_max = int(person_bb[2] * (1 + xis_factor_x))
        if mm_x_max > iw:
            mm_x_max = int(iw)
        roi_h = mm_y_max - mm_y_min
        roi_w = mm_x_max - mm_x_min
        roi_size = max(roi_h, roi_w)
        if roi_h > roi_w:
            roi_y_min = mm_y_min
            roi_y_max = mm_y_max
            roi_x_min = mm_x_min - int((roi_h - roi_w) / 2)
            roi_x_max = mm_x_max + int((roi_h - roi_w) / 2)
        elif roi_w > roi_h:
            roi_y_min = mm_y_min - int((roi_w - roi_h) / 2)
            roi_y_max = mm_y_max + int((roi_w - roi_h) / 2)
            roi_x_min = mm_x_min
            roi_x_max = mm_x_max
        else:
            roi_y_min = mm_y_min
            roi_y_max = mm_y_max
            roi_x_min = mm_x_min
            roi_x_max = mm_x_max
        if roi_y_min < 0:
            roi_y_min = 0
        if roi_y_max > ih:
            roi_y_max = int(ih)
        if roi_x_min < 0:
            roi_x_min = 0
        if roi_x_max > iw:
            roi_x_max = int(iw)
        # note : this may fail to be square (solve later)
        det_roi = in_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max, :]
        roi_bb = np.array([roi_x_min, roi_y_min, roi_x_max, roi_y_max, iw, ih])

        return det_roi, roi_bb

    def preprocess_img(self, in_img, in_bb):
        # apply ROI
        det_roi, roi_bb = self.handleroi(in_img, in_bb)
        # resize
        probe_img_0 = cv2.resize(det_roi, (self.input_size[1], self.input_size[2]),
                                 interpolation=cv2.INTER_AREA)
        probe_img_0 = cv2.cvtColor(probe_img_0, cv2.COLOR_BGR2RGB)
        probe_img_0 = np.array([probe_img_0]).astype(np.float32)
        # normalize
        # probe_img_0 -= 127.5
        # probe_img_0 /= 127.5
        probe_img_0 /= 255.
        # The original landmark model is expecting RGB [0, 1] frames.
        # https://github.com/geaxgx/depthai_blazepose/blob/c72a4a9652223a53cb63ea893810126ed3d63d12/README.md
        return probe_img_0

    def estimate(self, in_img, in_bb):
        probe_img = self.preprocess_img(in_img, in_bb)
        return self.worker.estimate(probe_img)


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# //////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------------------------------------------------------------------------
class TwoStage3D:
    def __init__(self, first_stage, second_stage, architecture):
        self._first_stage = first_stage
        self._second_stage = second_stage
        self._architecture = architecture
        self._setup()

    def _setup(self):
        self._est = PoseEstimation2D(method=self._first_stage, architecture=self._architecture)
        self._lif = PoseLifting(method=self._second_stage, architecture=self._architecture)

    def handleroi(self, in_img, in_bb):
        imginfo = in_img.shape
        iw = imginfo[0]
        ih = imginfo[1]
        # ms_bbox_result = in_bb
        # persons = ms_bbox_result[0]
        person_bb = in_bb[:4]
        person_bb = person_bb.astype(int)
        xis_factor_x = 0.05
        xis_factor_y = 0.05
        mm_y_min = int(person_bb[1] * (1 - xis_factor_y))
        if mm_y_min < 0:
            mm_y_min = 0
        mm_y_max = int(person_bb[3] * (1 + xis_factor_y))
        if mm_y_max > ih:
            mm_y_max = int(ih)
        mm_x_min = int(person_bb[0] * (1 - xis_factor_x))
        if mm_x_min < 0:
            mm_x_min = 0
        mm_x_max = int(person_bb[2] * (1 + xis_factor_x))
        if mm_x_max > iw:
            mm_x_max = int(iw)
        roi_h = mm_y_max - mm_y_min
        roi_w = mm_x_max - mm_x_min
        roi_size = max(roi_h, roi_w)
        if roi_h > roi_w:
            roi_y_min = mm_y_min
            roi_y_max = mm_y_max
            roi_x_min = mm_x_min - int((roi_h - roi_w) / 2)
            roi_x_max = mm_x_max + int((roi_h - roi_w) / 2)
        elif roi_w > roi_h:
            roi_y_min = mm_y_min - int((roi_w - roi_h) / 2)
            roi_y_max = mm_y_max + int((roi_w - roi_h) / 2)
            roi_x_min = mm_x_min
            roi_x_max = mm_x_max
        else:
            roi_y_min = mm_y_min
            roi_y_max = mm_y_max
            roi_x_min = mm_x_min
            roi_x_max = mm_x_max
        if roi_y_min < 0:
            roi_y_min = 0
        if roi_y_max > ih:
            roi_y_max = int(ih)
        if roi_x_min < 0:
            roi_x_min = 0
        if roi_x_max > iw:
            roi_x_max = int(iw)
        # note : this may fail to be square (solve later)
        det_roi = in_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max, :]
        roi_bb = np.array([roi_x_min, roi_y_min, roi_x_max, roi_y_max, iw, ih])

        return det_roi, roi_bb

    def preprocess_img(self, in_img, in_bb):
        # apply ROI
        det_roi, roi_bb = self.handleroi(in_img, in_bb)
        # resize
        probe_img_0 = cv2.resize(det_roi, (256, 256),  # hardcoded for hrnet todo: move all constants to somewhere else
                                 interpolation=cv2.INTER_AREA)
        probe_img_0 = cv2.cvtColor(probe_img_0, cv2.COLOR_BGR2RGB)
        probe_img_0 = np.array([probe_img_0]).astype(np.float32)
        # normalize
        # probe_img_0 -= 127.5
        # probe_img_0 /= 127.5
        # probe_img_0 /= 255.

        # input_i = np.array([probe_img_0, ], dtype=np.float32)
        input_i = np.moveaxis(probe_img_0, 3, 1)

        return roi_bb, input_i

    def preprocess_landmarks(self, in_pose, in_bb):
        recover_original_img_aspect_ratio = True
        if recover_original_img_aspect_ratio:
            bbd_width = in_bb[2] - in_bb[0]
            bbd_height = in_bb[3] - in_bb[1]

            o_pose = in_pose[0]
            _w = _h = 256.  # hardcoded hrnet out todo: move all constants to somewhere else

            in_pose_in = o_pose
            in_pose_in[:, 0] = o_pose[:, 0] / _w * bbd_width
            in_pose_in[:, 1] = o_pose[:, 1] / _h * bbd_height
            in_pose_in[:, 0] = in_pose_in[:, 0] + in_bb[0]
            in_pose_in[:, 1] = in_pose_in[:, 1] + in_bb[1]

            o_pose = (in_pose_in / self._ori_img_shape[0] * 2 - [1, self._ori_img_shape[1] / self._ori_img_shape[0]])
        else:
            o_pose = in_pose[0]
            in_pose_in = in_pose[0]
            _w = _h = 256.
            o_pose = (o_pose / _w * 2 - [1, _h / _w])

        return in_pose_in, o_pose

    def estimate(self, in_img, in_bb):
        self._ori_img_shape = in_img.shape
        rbb, t_img = self.preprocess_img(in_img, in_bb)
        t_td = self._est.estimate(t_img)
        #temptemp, t_tdp = self.preprocess_landmarks(t_td, in_bb) # wally
        temptemp, t_tdp = self.preprocess_landmarks(t_td, rbb)

        def drawPreds(image_in, preds_in):
            image_out = image_in.copy()

            for point in preds_in:
                #for point in set:
                x = int(point[0])
                y = int(point[1])
                image_out = cv2.circle(image_out, (x, y), 3, (255, 0, 0), thickness=-1)
                #(img, center, radius, color, thickness=None, lineType=None, shift=None):
            # cv2.imwrite('onetime.png', image_out)

            return image_out

        # asd = drawPreds(in_img, temptemp)
        t_op = self._lif.estimate(t_tdp)
        return t_op[0]


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# ----------------------------------------------------------------------------------------------
def test():
    pclass = PoseEstimation(method='hrnet_poseaugvpose', architecture='pytorch')
    wally = 'here'


# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # test()
    wally = 'here'

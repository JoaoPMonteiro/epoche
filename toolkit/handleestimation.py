#
import methodspaths
import torch
import importlib

# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////
#import sys
#sys.path.append('./')
try:
    import toolkit.customsnippets as customsnippets
except:
    import customsnippets
try:
    import toolkit.methodspaths as methodspaths
except:
    import methodspaths


class PoseEstimation:
    def __init__(self, method='blazepose_lite', architecture='tflite'):
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
                l_est = SingleStage3D(method=self._method, architecture=self._architecture)
            elif _selector == 1:
                x_mthd = self._method.split('_')
                l_est = TwoStage3D(first_stage=x_mthd[0], second_stage=x_mthd[1], architecture=self._architecture)
            else:
                raise TypeError('selection not valid')
        else:
            raise TypeError('method or architecture not valid')

        return 0

    def estimate(self, in_img):
        return 0


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////
class PoseEstimation2D:
    def __init__(self, method='hrnet', architecture='pytorch'):
        self._methods_pool = ['hrnet']
        self._architecture_pool = ['pytorch', 'onnx']
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

    def estimate(self, in_img):
        return 0


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////
class PoseLifting:
    def __init__(self, method='vpose', architecture='pytorch'):
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
            if id_arch[0] == 0:
                self._init_pytorch(id_method)
            elif id_arch[0] == 1:
                self._init_onnx()
            elif id_arch[0] == 2:
                self._init_vpu()
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
        wally = 'here'

    def _vpose(self):
        l_paths = self._paths
        evaluate = l_paths.pth
        x = l_paths.cfg
        x = x.replace('../', '')
        x = x.replace('/', '.')
        x = x.replace('.py', '')
        import sys

        sys.path.append('..')

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
        pass

    def _init_vpu(self):
        pass

    def estimate(self, in_img):
        return 0

    def get_model(self):
        return self._model


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# //////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------------------------------------------------------------------------
class SingleStage3D:
    def __init__(self, method='blazepose_lite', architecture='tflite'):
        self._methods_pool = ['blazepose_lite',
                              'blazepose_full',
                              'blazepose_heavy']
        self._architecture_pool = ['tflite', 'onnx']
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

    def estimate(self, in_img):
        return 0


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# //////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------------------------------------------------------------------------
class TwoStage3D:
    def __init__(self, first_stage='hrnet', second_stage='vpose', architecture='pytorch'):
        self._first_stage = first_stage
        self._second_stage = second_stage
        self._architecture = architecture
        self._setup()

    def _setup(self):
        pclass_1 = PoseEstimation2D(method=self._first_stage, architecture=self._architecture)
        pclass_2 = PoseLifting(method=self._second_stage, architecture=self._architecture)
        return 0

    def estimate(self, in_img):
        return 0


# ----------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////


# ----------------------------------------------------------------------------------------------
def test():
    pclass = PoseEstimation(method='hrnet_poseaugvpose', architecture='pytorch')
    wally = 'here'


# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    test()

# ----------------------------------------------------------------------------------------------------------------
from mmdet.models.detectors.yolox import YOLOX
from mmpose.models.detectors.top_down import TopDown
from mmpose.apis import init_pose_model
import torch

try:
    import toolkit.methodspaths as methodspaths
except:
    import methodspaths

import sys
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce


# ----------------------------------------------------------------------------------------------------------------
def build_custom_yolox_from_model(initialized_mmdet_model):
    _config = initialized_mmdet_model.cfg._cfg_dict['model']
    custom_yolox = CustomYOLOX(_config['backbone'],
                               _config['neck'],
                               _config['bbox_head'],
                               test_cfg=_config['test_cfg'])
    custom_yolox.forward = custom_yolox.customforward
    pth_file = methodspaths.methodsDict['yolox_Paths'].pth
    ckpt = torch.load(pth_file)
    custom_yolox.load_state_dict(ckpt['state_dict'])
    custom_yolox.eval()
    return custom_yolox


class CustomYOLOX(YOLOX):
    """this code is modified base on MMDET YOLOX code,
    https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox
    """

    @torch.no_grad()
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained, init_cfg)
        self.decodeBB = GetPoseDetectionBBNN()

    @torch.no_grad()
    def customforward(self, img):
        feat = self.extract_feat(img)
        outs = self.bbox_head(feat)
        ab = self.decodeBB(outs)
        return ab


# ----------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------
class GetLandMarksNet(nn.Module):
    def __init__(self):
        super(GetLandMarksNet, self).__init__()

    def _transform_preds(self, coords, center, scale, output_size):
        """Get final keypoint predictions from heatmaps and apply scaling and
        translation to map them back to the image.

        Note:
            num_keypoints: K

        Args:
            coords (np.ndarray[K, ndims]):

                * If ndims=2, corrds are predicted keypoint location.
                * If ndims=4, corrds are composed of (x, y, scores, tags)
                * If ndims=5, corrds are composed of (x, y, scores, tags,
                  flipped_tags)

            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            output_size (np.ndarray[2, ] | list(2,)): Size of the
                destination heatmaps.
            use_udp (bool): Use unbiased data processing

        Returns:
            np.ndarray: Predicted coordinates in the images.

        """
        use_udp = False

        # Recover the scale which is normalized by a factor of 200.
        scale = scale * 200.0

        if use_udp:
            scale_x = scale[0] / (output_size[0] - 1.0)
            scale_y = scale[1] / (output_size[1] - 1.0)
        else:
            scale_x = scale[0] / output_size[0]
            scale_y = scale[1] / output_size[1]

        target_coords_a = torch.ones(coords.shape[0], 1)
        target_coords_b = torch.ones(coords.shape[0], 1)
        target_coords_c = torch.ones(coords.shape[0], 2)

        target_coords_a = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords_b = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
        target_coords_a = target_coords_a.reshape(coords.shape[0], 1)
        target_coords_b = target_coords_b.reshape(coords.shape[0], 1)
        target_coords_c = torch.cat((target_coords_a, target_coords_b), 1)

        return target_coords_c

    def _get_max_preds(self, heatmaps):
        """Get keypoint predictions from score maps.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

        Returns:
            tuple: A tuple containing aggregated results.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        N, K, W = 1, 16, 64
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = torch.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))

        dubois = torch.tensor(0)
        for i, x in enumerate(heatmaps_reshaped[0]):
            c_probe = heatmaps_reshaped[0][i]
            bb = torch.max(c_probe)
            cc = bb.reshape(1, )

            if i == 0:
                dubois = cc
            else:
                dubois = torch.cat((dubois, cc), 0)

        # dubois = torch.max(heatmaps_reshaped, keepdim=False, dim=2).values

        maxvals = dubois.reshape(1, 16, 1)

        preds = torch.ones(N, K, 1)
        preds = torch.cat((idx, idx), 2).type(torch.float32)

        pkk = preds[:, :, 0]
        farc = preds[:, :, 1]

        just = pkk % float(W)
        just = just.reshape(16, 1)

        # jul = farc // float(W)
        jul = torch.div(farc, W, rounding_mode='floor')
        jul = jul.reshape(16, 1)
        ine = torch.cat((just, jul), 1)
        eta = ine.reshape(1, 16, 2)

        midlle = torch.cat((maxvals, maxvals), 2) > 0.0
        serag = torch.tensor([-1]).type(torch.float32)
        preds_o = torch.where(midlle, eta, serag[0])
        return preds_o, maxvals

    def _keypoints_from_heatmaps(self,
                                 heatmaps,
                                 center,
                                 scale  # ,
                                 # post_process='default'
                                 ):
        """Get final keypoint predictions from heatmaps and transform them back to
        the image.

        Note:
            batch size: N
            num keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
            center (np.ndarray[N, 2]): Center of the bounding box (x, y).
            scale (np.ndarray[N, 2]): Scale of the bounding box
                wrt height/width.
            post_process (str/None): Choice of methods to post-process
                heatmaps. Currently supported: None, 'default', 'unbiased',
                'megvii'.
            unbiased (bool): Option to use unbiased decoding. Mutually
                exclusive with megvii.
                Note: this arg is deprecated and unbiased=True can be replaced
                by post_process='unbiased'
                Paper ref: Zhang et al. Distribution-Aware Coordinate
                Representation for Human Pose Estimation (CVPR 2020).
            kernel (int): Gaussian kernel size (K) for modulation, which should
                match the heatmap gaussian sigma when training.
                K=17 for sigma=3 and k=11 for sigma=2.
            valid_radius_factor (float): The radius factor of the positive area
                in classification heatmap for UDP.
            use_udp (bool): Use unbiased data processing.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Classification target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).
                Paper ref: Huang et al. The Devil is in the Details: Delving into
                Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

        Returns:
            tuple: A tuple containing keypoint predictions and scores.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        # N, K, H, W = 1, 16, 64, 64
        N = torch.tensor(1)
        K = torch.tensor(16)
        H = torch.tensor(64)
        W = torch.tensor(64)

        preds, maxvals = self._get_max_preds(heatmaps)

        n = N - 1
        wally = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        # poljot = torch.tensor([-1,-1]).reshape(1,2)

        def func1(_preds, _n, _heatmap, _px, _py, _k, _poljot):
            eeeeeeeee = _heatmap.shape
            aaaaaaaaa = _heatmap[_py, _px + 1]
            bbbbbbbbb = _heatmap[_py, _px - 1]
            ccccccccc = _heatmap[_py + 1, _px]
            ddddddddd = _heatmap[_py - 1, _px]
            difparta = (aaaaaaaaa - bbbbbbbbb).view(1, )
            difpartb = (ccccccccc - ddddddddd).view(1, )
            diff = torch.cat((
                difparta,
                difpartb
            ))

            def sign_alt(in_tensor):
                sign_0 = torch.gt(in_tensor, 0.).float()
                sign_1 = torch.where(sign_0 == 0., torch.tensor(-1.), sign_0)
                sign_2 = torch.eq(in_tensor, 0.)
                sign_3 = ~sign_2
                sign_3 = sign_3.float()
                sign_alt = torch.mul(sign_1, sign_3)
                sign_alt = torch.where(sign_alt == -0., torch.tensor(0.), sign_alt)
                return sign_alt

            alpha = sign_alt(diff) * .25
            # alpha = torch.sign(diff) * .25
            beta = _preds[_n][_k]
            gamma = beta + alpha
            if _k == 0:
                _poljot = gamma.reshape(1, 2)
            else:
                _poljot = torch.cat((_poljot, gamma.reshape(1, 2)), 0)

            return _poljot

        def func2(_preds, _n, _heatmap, _px, _py, _k, _poljot):
            if _k == 0:
                omega = _preds[_n][_k]
                _poljot = omega.reshape(1, 2)
            else:
                omega = _preds[_n][_k]
                _poljot = torch.cat((_poljot, omega.reshape(1, 2)), 0)

            return _poljot

        poljot = torch.zeros(1, 1)
        for k, kk in enumerate(wally):
            heatmap = heatmaps[n][k]
            px = preds[n][k][0].int()
            py = preds[n][k][1].int()
            #px = torch.tensor(234567890, dtype=torch.int32)
            #py = torch.tensor(234567890, dtype=torch.int32)

            # _option0 = torch.zeros(1, dtype=torch.bool) # torch.where(px > torch.tensor(1.), True, False)
            # _option1 = torch.zeros(1, dtype=torch.bool) #torch.where(py > torch.tensor(1.), True, False)
            # _option2 = torch.zeros(1, dtype=torch.bool) #torch.where(torch.gt(torch.sub(W, 1.), px), True, False)
            # _option3 = torch.zeros(1, dtype=torch.bool) #torch.where(torch.gt(torch.sub(H, 1.), py), True, False)
            #_option0 = torch.where(px > 1.,
            #                       torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
            #_option1 = torch.where(py > 1.,
            #                       torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
            #_option2 = torch.where(torch.gt(W - 1, px),
            #                       torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
            #_option3 = torch.where(torch.gt(H - 1, py),
            #                       torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))

            #opone = torch.cat((_option0.view(1, ), _option1.view(1, )))
            #optwo = torch.cat((_option2.view(1, ), _option3.view(1, )))
            #op3 = torch.cat((opone, optwo))
            #op4 = torch.all(op3, 0, keepdim=False)

            backuptensor = torch.tensor(1, dtype=torch.int32)

            px_dummy = torch.where(px > 1.,
                                   px, backuptensor) #, dtype=torch.int32
            px_dummy = torch.where(torch.gt(W - 1, px_dummy),
                                   px_dummy, backuptensor)

            py_dummy = torch.where(py > 1.,
                                   py, backuptensor)
            py_dummy = torch.where(torch.gt(H - 1, py_dummy),
                                   py_dummy, backuptensor)

            #py_dummy = torch.where(op4 == torch.ones(1, dtype=torch.bool),
            #                       py,
            #                       torch.tensor(1, dtype=torch.int64))

            #px_dummy = px_dummy.long()
            #py_dummy = py_dummy.long()

            answer_a = func1(preds, n, heatmap, px_dummy, py_dummy, k, poljot)
            answer_b = func2(preds, n, heatmap, px_dummy, py_dummy, k, poljot)

            again0 = torch.where(px > 1.,
                                   1, 0)
            again1 = torch.where(py > 1.,
                                   1, 0)
            again2 = torch.where(torch.gt(W - 1, px),
                                   1, 0)
            again3 = torch.where(torch.gt(H - 1, py),
                                   1, 0)

            poljot = torch.where((again0 + again1 + again2 + again3)>3,
                                 answer_a,
                                 answer_b)
            # if 1 < px < W - 1 and 1 < py < H - 1:
            #if op4:
            #    poljot = answer_a
            #else:
            #    poljot = answer_b

        poljot = poljot.reshape(1, 16, 2)

        # Transform back to the image
        ind_i = N - 1
        outsize_t = torch.tensor([W, H])

        raketa = self._transform_preds(
            poljot[ind_i], center[ind_i], scale[ind_i], outsize_t)
        raketa = raketa.reshape(1, 16, 2)

        return raketa, maxvals

    def _get_new_seven(self, point_a_t, point_b_t):
        # https://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
        from numpy import ones, vstack
        from numpy.linalg import lstsq
        point_a = point_a_t.numpy()
        point_b = point_b_t.numpy()
        points = [point_a, point_b]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        new_y_height = abs(point_b[1] - point_a[1])
        if point_b[1] > point_a[1]:
            new_y = point_a[1] + new_y_height / 2.
        else:
            new_y = point_b[1] + new_y_height / 2.
        new_x = (new_y - c) / m

        return [np.round(new_x, 0), np.round(new_y, 0)]

    def _get_new_seven_alt(self, point_a_t, point_b_t):
        # https://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
        point_a_t = point_a_t.reshape(1, 2)
        point_b_t = point_b_t.reshape(1, 2)
        point_t = torch.cat((point_a_t, point_b_t), 0)
        out = torch.mean(point_t, 0)
        return torch.round(out)

    def _reorder_landmarks(self, input_landmarks):
        cheat_list = [6, 3, 4, 5, 2, 1, 0, 7, 8, 9, 12, 11, 10, 13, 14, 15]

        out_land_alt = input_landmarks[0][cheat_list]

        out_land = torch.ones(1, 2)
        for ii, jj in enumerate(cheat_list):
            if ii == 0:
                out_land = input_landmarks[0][jj, :].reshape(1, 2)
            elif ii == 7:
                n7 = self._get_new_seven_alt(out_land_alt[0, :].reshape(1, 2), out_land_alt[8, :].reshape(1, 2))
                out_land = torch.cat((out_land, n7.reshape(1, 2)), 0)
            else:
                out_land = torch.cat((out_land, input_landmarks[0][jj, :].reshape(1, 2)), 0)

        '''
        out_land_alt[7] = self._get_new_seven_alt(out_land_alt[0, :].reshape(1, 2), out_land_alt[8, :].reshape(1, 2))
        '''

        flip_h = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 13, 14, 15, 10, 11, 12]

        out_land = out_land[flip_h]
        return out_land.reshape(1, 16, 2)
        '''
        out_land = out_land_alt[flip_h]
        return out_land.reshape(1, 16, 2)
        '''

    def forward(self, heatmaps):
        c = torch.zeros(1, 2, dtype=torch.float32)
        center = torch.tensor([[128., 128.]]).type(torch.float32)
        c[0] = center[0]
        s = torch.tensor([[256.0 / 200, 256.0 / 200]], dtype=torch.float32)
        preds, maxvals = self._keypoints_from_heatmaps(heatmaps, c, s)
        predsh36m = self._reorder_landmarks(preds)

        return predsh36m


def build_custom_hrnet():
    in_cnfg = methodspaths.methodsDict['hrnet_Paths'].cfg
    in_chckpnt = methodspaths.methodsDict['hrnet_Paths'].pth
    model = init_pose_model(in_cnfg, in_chckpnt, device='cpu')
    _config = model.cfg._cfg_dict['model']
    custom_hrnet = CustomHRNET(_config['backbone'],
                               None,
                               _config['keypoint_head'],
                               _config['train_cfg'],
                               _config['test_cfg'],
                               _config['pretrained'],
                               None)
    custom_hrnet.forward = custom_hrnet.customforward
    ckpt = torch.load(in_chckpnt)
    custom_hrnet.load_state_dict(ckpt['state_dict'])

    return custom_hrnet


class CustomHRNET(TopDown):
    """
       this code is modified base on MMPOSE TOPDOWN HRNET code,
       https://github.com/open-mmlab/mmpose/blob/master/mmpose/models/detectors/top_down.py
    """

    @torch.no_grad()
    def __init__(self, *args):
        super().__init__(*args)
        self.decodeHM = GetLandMarksNet()

    @torch.no_grad()
    def customforward(self, img):
        img_n = torch.div(img, 255.)
        output = self.backbone(img_n)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        pose = self.decodeHM(output)
        return pose


# ----------------------------------------------------------------------------------------------------------------


x2 = methodspaths.methodsDict['gcn_Paths'].cfg
addit_path = "/".join(x2.split("/")[:-1])
sys.path.append(addit_path)
from sem_graph_conv import SemGraphConv
from sem_gcn import _GraphConv
from sem_gcn import _ResGraphConv
from sem_gcn import _GraphNonLocal
from sem_gcn import SemGCN


class CustomSemGraphConv(SemGraphConv):
    """this code is modified base on GCN code from the poseaug authors github,
    https://github.com/jfzhang95/PoseAug
    """

    def __init__(self, *args):
        super(CustomSemGraphConv, self).__init__(*args)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj, dtype=torch.float32).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output


class Custom_GraphConv(_GraphConv):
    """this code is modified base on GCN code from the poseaug authors github,
        https://github.com/jfzhang95/PoseAug
        """

    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(Custom_GraphConv, self).__init__(adj, input_dim, output_dim, p_dropout)

        self.gconv = CustomSemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class Custom_ResGraphConv(_ResGraphConv):
    """this code is modified base on GCN code from the poseaug authors github,
        https://github.com/jfzhang95/PoseAug
        """

    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(Custom_ResGraphConv, self).__init__(adj, input_dim, output_dim, hid_dim, p_dropout)

        self.gconv1 = Custom_GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = Custom_GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class CustomSemGCN(SemGCN):
    """this code is modified base on GCN code from the poseaug authors github,
        https://github.com/jfzhang95/PoseAug
        """

    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(CustomSemGCN, self).__init__(adj, hid_dim, coords_dim, num_layers, nodes_group, p_dropout)

        _gconv_input = [Custom_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(Custom_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(Custom_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = CustomSemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x):
        """
        input: bx16x2 / bx32
        output: bx16x3
        """
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 16, 2)
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out


# ----------------------------------------------------------------------------------------------------------------
x_stgcn = methodspaths.methodsDict['stgcn_Paths'].cfg
addit_path_stgcn = "/".join(x_stgcn.split("/")[:-1])
sys.path.append(addit_path_stgcn)
from st_gcn_single_frame_test import WrapSTGCN
from st_gcn_single_frame_test import Model

hardcodeddevice = 'cpu'


class CustomModel(Model):
    """this code is modified base on STGCN code from the poseaug authors github,
        https://github.com/jfzhang95/PoseAug
        """

    def __init__(self, *args):
        super(CustomModel, self).__init__(*args)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).to(hardcodeddevice)
        self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).to(hardcodeddevice)
        self.inter_channels = [128, 128, 256]
        self.fc_out = self.inter_channels[-1]

    def custom_get_im2col_indices(self):
        """
        """
        kt = torch.tensor([[0], [0], [0], [0], [0]])
        it = torch.tensor([[0], [0], [0], [0], [0]])
        jt = torch.tensor([[0], [1], [2], [3], [4]])

        return kt, it, jt

    def custom_im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
        """
            (adapted from numpy version from) https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
        """
        x_padded = torch.nn.functional.pad(x, (0, 0), mode='constant')
        k, i, j = self.custom_get_im2col_indices()
        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2)
        cols = cols.transpose(0, 2)
        cols = cols.reshape(field_height * field_width * C, -1)
        return cols

    def custom_maxPool(self, x_in):
        """
            (adapted from numpy version from) https://agustinus.kristia.de/techblog/2016/07/18/convnet-maxpool-layer/
        """
        x = x_in
        n = 1
        d = 256
        h = 1
        w = 5

        X_reshaped = x.reshape(n * d, 1, h, w)

        stride = 5
        X_col = self.custom_im2col_indices(X_reshaped, h, w, padding=0, stride=stride)

        max_idx = torch.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.shape[0])]

        h_out = 1
        w_out = 1
        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 3)
        out = out.transpose(2, 0)
        out = out.transpose(0, 1)

        return out

    def forward(self, x, out_all_frame=False):

        # data normalization
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N, M, V, C, T
        x = x.view(N * M, V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, 1, -1)  # (N * M), C, 1, (T*V)

        # forwad GCN
        gcn_list = list(self.st_gcn_networks)
        for i_gcn, gcn in enumerate(gcn_list):
            x, _ = gcn(x, self.A)  # (N * M), C, 1, (T*V)

        x = x.view(N, -1, T, V)  # N, C, T ,V

        # --
        x_i_0 = x[:, :, :, self.graph.part[0]]
        x_i_0 = self.graph_max_pool(x_i_0, (1, 3))
        x_sub1_alt = x_i_0
        x_i_1 = x[:, :, :, self.graph.part[1]]
        x_i_1 = self.graph_max_pool(x_i_1, (1, 3))
        x_sub1_alt = torch.cat((x_sub1_alt, x_i_1), -1)
        x_i_2 = x[:, :, :, self.graph.part[2]]
        x_i_2 = self.graph_max_pool(x_i_2, (1, 3))
        x_sub1_alt = torch.cat((x_sub1_alt, x_i_2), -1)
        x_i_3 = x[:, :, :, self.graph.part[3]]
        x_i_3 = self.graph_max_pool(x_i_3, (1, 3))
        x_sub1_alt = torch.cat((x_sub1_alt, x_i_3), -1)
        x_i_4 = x[:, :, :, self.graph.part[4]]
        x_i_4 = self.custom_maxPool(x_i_4)
        x_sub1_alt = torch.cat((x_sub1_alt, x_i_4), -1)
        x_sub1 = x_sub1_alt
        # --

        x_sub1, _ = self.st_gcn_pool[0](x_sub1.view(N, -1, 1, T * len(self.graph.part)),
                                        self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
        x_sub1, _ = self.st_gcn_pool[1](x_sub1, self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
        x_sub1 = x_sub1.view(N, -1, T, len(self.graph.part))

        x_pool_1 = self.graph_max_pool(x_sub1, (1, len(self.graph.part)))  # N, 512, T, 1
        x_pool_1 = self.conv4(x_pool_1)  # N, C, T, 1

        x_up_sub = torch.cat((x_pool_1.repeat(1, 1, 1, len(self.graph.part)), x_sub1), 1)  # N, 1024, T, 5
        x_up_sub = self.conv2(x_up_sub)  # N, C, T, 5

        # --
        # upsample
        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_wally = x_up_sub[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node)
            x_wally_sub1 = torch.cat((x_wally_sub1, x_wally), -1) if i > 0 else x_wally

        qwer_a = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12], [13, 14, 15, 16]]

        tmeptemp_0 = x_wally_sub1[:, :, :, qwer_a[0]]  # [11, 12, 13]
        tmeptemp_1 = x_wally_sub1[:, :, :, qwer_a[1]]  # [14, 15, 16]
        tmeptemp_2 = x_wally_sub1[:, :, :, qwer_a[2]]  # [4, 5, 6]
        tmeptemp_3 = x_wally_sub1[:, :, :, qwer_a[3]]  # [1, 2, 3]
        tmeptemp_4 = x_wally_sub1[:, :, :, qwer_a[4]]  # [0]
        tmeptemp_5 = x_wally_sub1[:, :, :, qwer_a[5]]  # [7, 8, 9, 10]

        tempWally_0 = torch.cat((tmeptemp_4, tmeptemp_3), -1)
        tempWally_0 = torch.cat((tempWally_0, tmeptemp_2), -1)
        tempWally_0 = torch.cat((tempWally_0, tmeptemp_5), -1)
        tempWally_0 = torch.cat((tempWally_0, tmeptemp_0), -1)
        tempWally_0 = torch.cat((tempWally_0, tmeptemp_1), -1)

        x = torch.cat((x, tempWally_0), 1)
        # --
        x = self.non_local(x)  # N, 2C, T, V
        x = self.fcn(x)  # N, 3, T, V

        # output
        x = x.view(N, M, -1, T, V).permute(0, 2, 3, 4, 1).contiguous()  # N, C, T, V, M
        if out_all_frame:
            x_out = x
        else:
            x_out = x[:, :, self.pad].unsqueeze(2)
        return x_out


class CustomWrapSTGCN(WrapSTGCN):
    """this code is modified base on STGCN code from the poseaug authors github,
            https://github.com/jfzhang95/PoseAug
            """

    def __init__(self, p_dropout):
        super(CustomWrapSTGCN, self).__init__(p_dropout)
        self.stgcn = CustomModel(p_dropout)

    def forward(self, x):
        """
        input: bx16x2 / bx32
        output: bx16x3
        """
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 16, 2)
        # add one joint: 16 to 17
        Ct = torch.Tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]).transpose(1, 0)

        Ct = Ct.to(x.device)
        C = Ct.repeat([x.size(0), 1, 1]).view(-1, 16, 17)
        x = x.view(x.size(0), -1, 2)  # nx16x2
        x = x.permute(0, 2, 1).contiguous()  # nx2x16
        x = torch.matmul(x, C)  # nx2x17

        # process to stgcn
        x = x.unsqueeze(2).unsqueeze(-1)  # nx2x1x17x1
        out = self.stgcn(x)  # nx3x1x17x1
        out = out.view((1, 3, 17))

        # remove the joint: 17 to 16
        Ct17 = torch.Tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]).transpose(1, 0)
        Ct17 = Ct17.to(out.device)
        C17 = Ct17.repeat([out.size(0), 1, 1]).view(-1, 17, 16)
        out = torch.matmul(out, C17)  # nx2x17

        out = out.permute(0, 2, 1).contiguous()  # nx16x3
        return out


# ----------------------------------------------------------------------------------------------------------------
# post process yolox detecion based on the code from https://github.com/Megvii-BaseDetection/YOLOX/
class GetPoseDetectionBBNN(nn.Module):
    def __init__(self):
        super(GetPoseDetectionBBNN, self).__init__()

    def _dummytokeep(self, valid_boxes, valid_scores, nms_thr):
        sorted = torch.sort(valid_scores, descending=True)[1]

        return sorted

    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.sort(scores, descending=True)[1]

        keep = []
        while order.size()[0] > 0:
            i = order[0]
            keep.append(i)
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = torch.maximum(torch.tensor([0.0]), xx2 - xx1 + 1)
            h = torch.maximum(torch.tensor([0.0]), yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = torch.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def _multiclass_nms_class_agnostic(self, _boxes, _scores):
        """
        Multiclass NMS implemented in Numpy. Class-agnostic version.
        https://github.com/Megvii-BaseDetection/YOLOX/blob/25f116b3145db3f2808ffbe722677d301e34f40d/yolox/utils/demo_utils.py#L1

        :param scores:
        :param nms_thr:
        :return:
        """
        score_thr = 0.1

        left = torch.tensor([320.0, 320.0, 540.0, 540.0]).reshape(1, 4)
        right_1 = torch.tensor([0.46]).reshape(1, 1)
        right_2 = torch.zeros(1, 79)
        right = torch.cat((right_1, right_2), 1)

        _left = torch.tensor([0.0, 0.0, 640.0, 640.0]).reshape(1, 4)
        _right_1 = torch.tensor([0.47]).reshape(1, 1)
        _right_2 = torch.zeros(1, 79)
        _right = torch.cat((_right_1, _right_2), 1)

        boxes__ = torch.cat((_boxes, _left), 0)
        scores__ = torch.cat((_scores, _right), 0)

        boxes = torch.cat((boxes__, left), 0)
        scores = torch.cat((scores__, right), 0)

        cls_inds = scores.argmax(1)
        cls_scores = scores[torch.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr

        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]

        # keep = self.nms(valid_boxes, valid_scores, 0.45)
        keep = self._dummytokeep(valid_boxes, valid_scores, 0.45)

        dets_a = valid_boxes[keep, :]
        dets_b = valid_scores[keep, None]
        dets_c = valid_cls_inds[keep, None].float()
        dets = torch.cat((dets_a, dets_b, dets_c), 1)
        return dets

    def _demo_postprocess(self, outputs):
        """
        https://github.com/Megvii-BaseDetection/YOLOX/blob/25f116b3145db3f2808ffbe722677d301e34f40d/yolox/utils/demo_utils.py#L1

        :param outputs:
        :return:
        """
        strides = torch.tensor([8, 16, 32])

        img_size = torch.tensor([640, 640])

        # hsizes = [img_size[0] // stride for stride in strides]
        # wsizes = [img_size[1] // stride for stride in strides]
        hsizes = [torch.div(img_size[0], stride, rounding_mode='trunc') for stride in strides]
        wsizes = [torch.div(img_size[1], stride, rounding_mode='trunc') for stride in strides]

        shapes_hard = [torch.ones(1, 6400), torch.ones(1, 1600), torch.ones(1, 400)]

        # iter zero
        hsize0, wsize0, stride0, shapeF0 = hsizes[0], wsizes[0], strides[0], shapes_hard[0]
        xv0, yv0 = torch.meshgrid(torch.arange(wsize0), torch.arange(hsize0))
        xv0 = torch.permute(xv0, (1, 0))
        yv0 = torch.permute(yv0, (1, 0))
        grid0 = torch.stack((xv0, yv0), 2).reshape(1, -1, 2)
        shape0 = shapeF0.shape
        grids = grid0
        badu0 = torch.ones(shape0[0], shape0[1], 1) * stride0
        expanded_strides = badu0

        # iter one
        hsize1, wsize1, stride1, shapeF1 = hsizes[1], wsizes[1], strides[1], shapes_hard[1]
        xv1, yv1 = torch.meshgrid(torch.arange(wsize1), torch.arange(hsize1))
        xv1 = torch.permute(xv1, (1, 0))
        yv1 = torch.permute(yv1, (1, 0))
        grid1 = torch.stack((xv1, yv1), 2).reshape(1, -1, 2)
        shape1 = shapeF1.shape
        grids = torch.cat((grids, grid1), 1)
        badu1 = torch.ones(shape1[0], shape1[1], 1) * stride1
        expanded_strides = torch.cat((expanded_strides, badu1), 1)

        # iter teo
        hsize2, wsize2, stride2, shapeF2 = hsizes[2], wsizes[2], strides[2], shapes_hard[2]
        xv2, yv2 = torch.meshgrid(torch.arange(wsize2), torch.arange(hsize2))
        xv2 = torch.permute(xv2, (1, 0))
        yv2 = torch.permute(yv2, (1, 0))
        grid2 = torch.stack((xv2, yv2), 2).reshape(1, -1, 2)
        shape2 = shapeF2.shape
        grids = torch.cat((grids, grid2), 1)
        badu2 = torch.ones(shape2[0], shape2[1], 1) * stride2
        expanded_strides = torch.cat((expanded_strides, badu2), 1)

        outs_alpha = (outputs[..., :2] + grids) * expanded_strides
        outs_beta = torch.exp(outputs[..., 2:4]) * expanded_strides
        outs_gamma = outputs[..., 4:]
        outs_2 = torch.cat((outs_alpha, outs_beta, outs_gamma), 2)

        return outs_2

    def _postprocess_results_yolox(self, in_data):
        """
        https://github.com/Megvii-BaseDetection/YOLOX/

        :param in_data:
        :return:
        """
        predictions = self._demo_postprocess(in_data)[0]

        ratio = torch.tensor([1.0])

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy_a = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy_a = boxes_xyxy_a.reshape(boxes.shape[0], 1)

        boxes_xyxy_b = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy_b = boxes_xyxy_b.reshape(boxes.shape[0], 1)

        boxes_xyxy_c = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy_c = boxes_xyxy_c.reshape(boxes.shape[0], 1)

        boxes_xyxy_d = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy_d = boxes_xyxy_d.reshape(boxes.shape[0], 1)

        boxes_xyxy = torch.cat((boxes_xyxy_a, boxes_xyxy_b, boxes_xyxy_c, boxes_xyxy_d), 1)
        boxes_xyxy /= ratio

        dets = self._multiclass_nms_class_agnostic(boxes_xyxy, scores)

        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

        wpeople = (final_cls_inds == 0.0).nonzero()

        people = wpeople[0]

        lout = final_boxes[people]

        fso = final_scores[people].reshape(1, 1)
        lout = torch.cat((lout, fso), 1)

        return lout

    def forward(self, in_data):
        cls_scores, bbox_preds, objectnesses = in_data

        num_imgs = 1
        cls_out_channels = 80
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()

        # scale_factors = torch.tensor([639./1000., 640./1002., 639./1000., 640./1002.])
        # flatten_bbox_preds[..., :4] /= flatten_bbox_preds.new_tensor(
        #    scale_factors).unsqueeze(1)

        c = flatten_objectness.view(1, 8400, 1)
        ab = torch.cat((flatten_bbox_preds, c), 2)
        ab = torch.cat((ab, flatten_cls_scores), 2)
        return self._postprocess_results_yolox(ab)

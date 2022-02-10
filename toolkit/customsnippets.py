# ----------------------------------------------------------------------------------------------------------------
from mmdet.models.detectors.yolox import YOLOX
import torch


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

try:
    import toolkit.methodspaths as methodspaths
except:
    import methodspaths

import sys
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce

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

        #keep = self.nms(valid_boxes, valid_scores, 0.45)
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

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]
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

        #scale_factors = torch.tensor([639./1000., 640./1002., 639./1000., 640./1002.])
        #flatten_bbox_preds[..., :4] /= flatten_bbox_preds.new_tensor(
        #    scale_factors).unsqueeze(1)

        c = flatten_objectness.view(1, 8400, 1)
        ab = torch.cat((flatten_bbox_preds, c), 2)
        ab = torch.cat((ab, flatten_cls_scores), 2)
        return self._postprocess_results_yolox(ab)

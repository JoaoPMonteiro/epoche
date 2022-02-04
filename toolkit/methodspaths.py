class MethodsPaths:
    def __init__(self,
                 path_to_pth='',
                 path_to_cfg='',
                 path_to_onnx='',
                 path_to_blob='',
                 path_to_tflite=''
                 ):
        self.pth = path_to_pth
        self.cfg = path_to_cfg
        self.onnx = path_to_onnx
        self.blob = path_to_blob
        self.tflite = path_to_tflite


methodsDict = {
# blazepose -----------------------------------------------------------------------------------------------------------
# lite ----------------------
'blazepose_lite_Paths' : MethodsPaths(path_to_tflite='../methods/BLAZEPOSE/tflite_files/pose_landmark_lite_v084.tflite',
                                    path_to_blob='../methods/BLAZEPOSE/blob_files/pose_landmark_lite_v084_openvino_2021.4_4shave.blob',
                                    path_to_onnx='../methods/BLAZEPOSE/onnx_files/pose_landmark_lite_v084.onnx'),

# heavy ---------------------
'blazepose_heavy_Paths' : MethodsPaths(path_to_tflite='../methods/BLAZEPOSE/tflite_files/pose_landmark_heavy_v084.tflite',
                                     path_to_blob='../methods/BLAZEPOSE/blob_files/pose_landmark_heavy_v084_openvino_2021.4_4shave.blob',
                                     path_to_onnx='../methods/BLAZEPOSE/onnx_files/pose_landmark_heavy_v084.onnx'),

# full ----------------------
'blazepose_full_Paths' : MethodsPaths(path_to_tflite='../methods/BLAZEPOSE/tflite_files/pose_landmark_full_v084.tflite',
                                    path_to_blob='../methods/BLAZEPOSE/blob_files/pose_landmark_full_v084_openvino_2021.4_4shave.blob',
                                    path_to_onnx='../methods/BLAZEPOSE/onnx_files/pose_landmark_full_v084.onnx'),

# yolox ---------------------------------------------------------------------------------------------------------------
'yolox_Paths' : MethodsPaths(path_to_pth='../methods/YOLOX/pth_files/yolox_tiny_8x8_300e_coco.pth',
                           path_to_cfg='../methods/YOLOX/config_files/yolox_tiny_8x8_300e_coco.py',
                           path_to_blob='../methods/YOLOX/blob_files/yolox_tiny_8x8_300e_coco_openvino_2021.4_4shave.blob',
                           path_to_onnx='../methods/YOLOX/onnx_files/yolox_tiny_8x8_300e_coco.onnx'),

# hrnet ---------------------------------------------------------------------------------------------------------------
'hrnet_Paths' : MethodsPaths(path_to_pth='../methods/HRNET/pth_files/hrnet_w32_mpii_256x256.pth',
                           path_to_cfg='../methods/HRNET/config_files/hrnet_w32_mpii_256x256.py',
                           path_to_blob='../methods/HRNET/blob_files/hrnet_w32_mpii_256x256_openvino_2021.4_4shave.blob',
                           path_to_onnx='../methods/HRNET/onnx_files/hrnet_w32_mpii_256x256.onnx'),

# vpose ---------------------------------------------------------------------------------------------------------------
'vpose_Paths' : MethodsPaths(path_to_pth='../methods/VPOSE/pth_files/ckpt_best.pth.tar',
                           path_to_blob='../methods/VPOSE/blob_files/vpose_baseline_openvino_2021.4_4shave.blob',
                           path_to_onnx='../methods/VPOSE/onnx_files/vpose_baseline.onnx',
                           path_to_cfg='../methods/VPOSE/config_files/model_VideoPose3D.py'),

# sbl -----------------------------------------------------------------------------------------------------------------
'sbl_Paths' : MethodsPaths(path_to_pth='../methods/SBL/pth_files/ckpt_best.pth.tar',
                         path_to_blob='../methods/SBL/blob_files/sbl_baseline_openvino_2021.4_4shave.blob',
                         path_to_onnx='../methods/SBL/onnx_files/sbl_baseline.onnx',
                         path_to_cfg='../methods/SBL/config_files/linear_model.py'),

# gcn -----------------------------------------------------------------------------------------------------------------
'gcn_Paths' : MethodsPaths(path_to_pth='../methods/GCN/pth_files/ckpt_best.pth.tar',
                         path_to_blob='../methods/GCN/blob_files/gcn_baseline_openvino_2021.4_4shave.blob',
                         path_to_onnx='../methods/GCN/onnx_files/gcn_baseline.onnx',
                         path_to_cfg='../methods/GCN/config_files/sem_gcn.py'),

# stgcn ---------------------------------------------------------------------------------------------------------------
'stgcn_Paths' : MethodsPaths(path_to_pth='../methods/STGCN/pth_files/ckpt_best.pth.tar',
                           path_to_blob='../methods/STGCN/blob_files/stgcn_baseline_openvino_2021.4_4shave.blob',
                           path_to_onnx='../methods/STGCN/onnx_files/stgcn_baseline.onnx',
                           path_to_cfg='../methods/STGCN/config_files/st_gcn_single_frame_test.py'),

# poseaug -------------------------------------------------------------------------------------------------------------
# vpose ----------------------
'poseaugvpose_Paths' : MethodsPaths(path_to_pth='../methods/VPOSE/POSEAUG/pth_files/ckpt_best_dhp_p1.pth.tar',
                                  path_to_blob='../methods/VPOSE/POSEAUG/blob_files/vpose_poseaug_openvino_2021.4_4shave.blob',
                                  path_to_onnx='../methods/VPOSE/POSEAUG/onnx_files/vpose_poseaug.onnx',
                                  path_to_cfg='../methods/VPOSE/config_files/model_VideoPose3D.py'),

# sbl ------------------------
'poseaugsbl_Paths' : MethodsPaths(path_to_pth='../methods/SBL/POSEAUG/pth_files/ckpt_best_dhp_p1.pth.tar',
                                path_to_blob='../methods/SBL/POSEAUG/blob_files/sbl_poseaug_openvino_2021.4_4shave.blob',
                                path_to_onnx='../methods/SBL/POSEAUG/onnx_files/sbl_poseaug.onnx',
                                path_to_cfg='../methods/SBL/config_files/linear_model.py'),

# gcn ------------------------
'poseauggcn_Paths' : MethodsPaths(path_to_pth='../methods/GCN/POSEAUG/pth_files/ckpt_best_dhp_p1.pth.tar',
                                path_to_blob='../methods/GCN/POSEAUG/blob_files/gcn_poseaug_openvino_2021.4_4shave.blob',
                                path_to_onnx='../methods/GCN/POSEAUG/onnx_files/gcn_poseaug.onnx',
                                path_to_cfg='../methods/GCN/config_files/sem_gcn.py'),

# stgcn ----------------------
'poseaugstgcn_Paths' : MethodsPaths(path_to_pth='../methods/STGCN/POSEAUG/pth_files/ckpt_best_dhp_p1.pth.tar',
                                  path_to_blob='../methods/STGCN/POSEAUG/blob_files/stgcn_poseaug_openvino_2021.4_4shave.blob',
                                  path_to_onnx='../methods/STGCN/POSEAUG/onnx_files/stgcn_poseaug.onnx',
                                  path_to_cfg='../methods/STGCN/config_files/st_gcn_single_frame_test.py')
}
import wget
from os import path
import os


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)


def get_yolox_from_mmdet():
    #url_pth = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
    url_pth = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
    url_config_0 = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/f08548bfd6d394a82566022709b5ce9e6b0a855e/configs/yolox/yolox_s_8x8_300e_coco.py'
    #url_config_1 = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/f08548bfd6d394a82566022709b5ce9e6b0a855e/configs/yolox/yolox_tiny_8x8_300e_coco.py'
    url_config_2 = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/f08548bfd6d394a82566022709b5ce9e6b0a855e/configs/_base_/default_runtime.py'
    url_config_3 = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/f08548bfd6d394a82566022709b5ce9e6b0a855e/configs/_base_/schedules/schedule_1x.py'

    path_pth = 'methods/YOLOX/pth_files'
    if not os.path.isdir(path_pth):
        os.makedirs(path_pth)

    path_config = 'methods/YOLOX/config_files'
    if not os.path.isdir(path_config):
        os.makedirs(path_config)

    filename_pth = 'yolox_s_8x8_300e_coco.pth'
    filename_config_0 = 'yolox_s_8x8_300e_coco.py'
    #filename_config_1 = 'yolox_tiny_8x8_300e_coco.py'
    filename_config_2 = 'default_runtime.py'
    filename_config_3 = 'schedule_1x.py'

    out_pth = path.join(path_pth, filename_pth)
    out_config_0 = path.join(path_config, filename_config_0)
    #out_config_1 = path.join(path_config, filename_config_1)
    out_config_2 = path.join(path_config, filename_config_2)
    out_config_3 = path.join(path_config, filename_config_3)

    print('downloading '+filename_pth+'...')
    wget.download(url_pth, out_pth)
    print('\ndone')
    print('downloading ' + filename_config_0 + '...')
    wget.download(url_config_0, out_config_0)
    print('\ndone')
    #print('downloading ' + filename_config_1 + '...')
    #wget.download(url_config_1, out_config_1)
    #print('\ndone')
    print('downloading ' + filename_config_2 + '...')
    wget.download(url_config_2, out_config_2)
    print('\ndone')
    print('downloading ' + filename_config_3 + '...')
    wget.download(url_config_3, out_config_3)
    print('\ndone')

    new_string = './'
    inplace_change(out_config_0, '../_base_/schedules/', new_string)
    inplace_change(out_config_0, '../_base_/', new_string)


def get_hrnet_from_mmpose():
    url_pth = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256-6c4f923f_20200812.pth'
    url_config_0 = 'https://raw.githubusercontent.com/open-mmlab/mmpose/dca589a0388530d4e387d1200744ad35dd30768d/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hrnet_w32_mpii_256x256.py'
    url_config_1 = 'https://raw.githubusercontent.com/open-mmlab/mmpose/dca589a0388530d4e387d1200744ad35dd30768d/configs/_base_/datasets/mpii.py'

    path_pth = 'methods/HRNET/pth_files'
    if not os.path.isdir(path_pth):
        os.makedirs(path_pth)

    path_config = 'methods/HRNET/config_files'
    if not os.path.isdir(path_config):
        os.makedirs(path_config)

    filename_pth = 'hrnet_w32_mpii_256x256.pth'
    filename_config_0 = 'hrnet_w32_mpii_256x256.py'
    filename_config_1 = 'mpii.py'

    out_pth = path.join(path_pth, filename_pth)
    out_config_0 = path.join(path_config, filename_config_0)
    out_config_1 = path.join(path_config, filename_config_1)

    print('downloading '+filename_pth+'...')
    wget.download(url_pth, out_pth)
    print('\ndone')
    print('downloading ' + filename_config_0 + '...')
    wget.download(url_config_0, out_config_0)
    print('\ndone')
    print('downloading ' + filename_config_1 + '...')
    wget.download(url_config_1, out_config_1)
    print('\ndone')

    oldString = '../../../../_base_/datasets/mpii.py'
    newString = 'mpii.py'
    inplace_change(out_config_0, oldString, newString)


def get_poseaug_from_authorsgit():
    import gdown

    url_poseaug_gcn = 'https://drive.google.com/drive/folders/1uzFruLfzc9pPrtL-zDVotEVJOfnHs0Gc'
    url_poseaug_mlp = 'https://drive.google.com/drive/folders/1BnGlyPZ4_rmyLF178z_LU6H-JvXf89T-'
    url_poseaug_stgcn = 'https://drive.google.com/drive/folders/1vfAKAVFsVw0aCrxT7Ranl7eRegum0DPW'
    url_poseaug_videopose = 'https://drive.google.com/drive/folders/1fYh68cwulC65Ry_RDt-YwepirXBh5KqG'
    
    url_gcn = 'https://drive.google.com/drive/folders/1n_ubu5lgN5weyKiPsFB9LI6Oig5ywN1k'
    url_mlp = 'https://drive.google.com/drive/folders/1czwx0vueiRBxsQ63Ki2WH04ZWb4x_df0'
    url_stgcn = 'https://drive.google.com/drive/folders/1nLrem6QrH_VEZQiYYa05_zM2v3SxGGek'
    url_videopose = 'https://drive.google.com/drive/folders/12QG1qhSqTPw7qHQyNXaKXpoebg99buxI'

    path_pth_0_0 = 'methods/GCN/POSEAUG/pth_files'
    path_pth_0_1 = 'methods/STGCN/POSEAUG/pth_files'
    path_pth_0_2 = 'methods/SBL/POSEAUG/pth_files'
    path_pth_0_3 = 'methods/VPOSE/POSEAUG/pth_files'
    path_pth_1 = 'methods/GCN/pth_files'
    path_pth_2 = 'methods/STGCN/pth_files'
    path_pth_3 = 'methods/SBL/pth_files'
    path_pth_4 = 'methods/VPOSE/pth_files'

    if not os.path.isdir(path_pth_0_0):
        os.makedirs(path_pth_0_0)
    if not os.path.isdir(path_pth_0_1):
        os.makedirs(path_pth_0_1)
    if not os.path.isdir(path_pth_0_2):
        os.makedirs(path_pth_0_2)
    if not os.path.isdir(path_pth_1):
        os.makedirs(path_pth_1)
    if not os.path.isdir(path_pth_2):
        os.makedirs(path_pth_2)
    if not os.path.isdir(path_pth_3):
        os.makedirs(path_pth_3)
    if not os.path.isdir(path_pth_4):
        os.makedirs(path_pth_4)

    print('downloading ' + path_pth_0_0 + '...')
    gdown.download_folder(url_poseaug_gcn, output=path_pth_0_0, quiet=True)
    print('\ndone')
    print('downloading ' + path_pth_0_1 + '...')
    gdown.download_folder(url_poseaug_stgcn, output=path_pth_0_1, quiet=True)
    print('\ndone')
    print('downloading ' + path_pth_0_2 + '...')
    gdown.download_folder(url_poseaug_mlp, output=path_pth_0_2, quiet=True)
    print('\ndone')
    print('downloading ' + path_pth_0_3 + '...')
    gdown.download_folder(url_poseaug_videopose, output=path_pth_0_3, quiet=True)
    print('\ndone')
    print('downloading ' + path_pth_1 + '...')
    gdown.download_folder(url_gcn, output=path_pth_1, quiet=True)
    print('\ndone')
    print('downloading ' + path_pth_2 + '...')
    gdown.download_folder(url_stgcn, output=path_pth_2, quiet=True)
    print('\ndone')
    print('downloading ' + path_pth_3 + '...')
    gdown.download_folder(url_mlp, output=path_pth_3, quiet=True)
    print('\ndone')
    print('downloading ' + path_pth_4 + '...')
    gdown.download_folder(url_videopose, output=path_pth_4, quiet=True)
    print('\ndone')

    path_pth_5 = 'methods/GCN/config_files'
    path_pth_6 = 'methods/STGCN/config_files'
    path_pth_7 = 'methods/SBL/config_files'
    path_pth_8 = 'methods/VPOSE/config_files'

    if not os.path.isdir(path_pth_5):
        os.makedirs(path_pth_5)
    if not os.path.isdir(path_pth_6):
        os.makedirs(path_pth_6)
    if not os.path.isdir(path_pth_7):
        os.makedirs(path_pth_7)
    if not os.path.isdir(path_pth_8):
        os.makedirs(path_pth_8)

    url_vpose_0 = 'https://raw.githubusercontent.com/jfzhang95/PoseAug/f3f5c4e916ebf7529b873ec1c14c1ce0bf0f5cb1/models_baseline/videopose/model_VideoPose3D.py'
    vpose_out_0 = path.join(path_pth_8, 'model_VideoPose3D.py')

    print('downloading ' + path_pth_8 + '...')
    wget.download(url_vpose_0, vpose_out_0)
    print('\ndone')

    url_sbl_0 = 'https://raw.githubusercontent.com/jfzhang95/PoseAug/f3f5c4e916ebf7529b873ec1c14c1ce0bf0f5cb1/models_baseline/mlp/linear_model.py'
    sbl_sbl_0 = path.join(path_pth_7, 'linear_model.py')

    print('downloading ' + path_pth_7 + '...')
    wget.download(url_sbl_0, sbl_sbl_0)
    print('\ndone')

    url_stgcn_0 = 'https://raw.githubusercontent.com/jfzhang95/PoseAug/f3f5c4e916ebf7529b873ec1c14c1ce0bf0f5cb1/models_baseline/models_st_gcn/st_gcn_single_frame_test.py'
    url_stgcn_1 = 'https://raw.githubusercontent.com/jfzhang95/PoseAug/f3f5c4e916ebf7529b873ec1c14c1ce0bf0f5cb1/models_baseline/models_st_gcn/st_gcn_utils/tgcn.py'
    url_stgcn_2 = 'https://raw.githubusercontent.com/jfzhang95/PoseAug/f3f5c4e916ebf7529b873ec1c14c1ce0bf0f5cb1/models_baseline/models_st_gcn/st_gcn_utils/graph_frames.py'
    url_stgcn_3 = 'https://raw.githubusercontent.com/jfzhang95/PoseAug/f3f5c4e916ebf7529b873ec1c14c1ce0bf0f5cb1/models_baseline/models_st_gcn/st_gcn_utils/graph_frames_withpool_2.py'
    url_stgcn_4 = 'https://raw.githubusercontent.com/jfzhang95/PoseAug/f3f5c4e916ebf7529b873ec1c14c1ce0bf0f5cb1/models_baseline/models_st_gcn/st_gcn_utils/st_gcn_non_local_embedded_gaussian.py'

    stgcn_out_0 = path.join(path_pth_6, 'st_gcn_single_frame_test.py')
    stgcn_out_1 = path.join(path_pth_6, 'tgcn.py')
    stgcn_out_2 = path.join(path_pth_6, 'graph_frames.py')
    stgcn_out_3 = path.join(path_pth_6, 'graph_frames_withpool_2.py')
    stgcn_out_4 = path.join(path_pth_6, 'st_gcn_non_local_embedded_gaussian.py')

    print('downloading ' + path_pth_6 + '...')
    wget.download(url_stgcn_0, stgcn_out_0)
    wget.download(url_stgcn_1, stgcn_out_1)
    wget.download(url_stgcn_2, stgcn_out_2)
    wget.download(url_stgcn_3, stgcn_out_3)
    wget.download(url_stgcn_4, stgcn_out_4)
    print('\ndone')

    oldString = 'from models_baseline.models_st_gcn.st_gcn_utils.'
    newString = 'from '
    inplace_change(stgcn_out_0, oldString, newString)

    url_gcn_0 = 'https://raw.githubusercontent.com/jfzhang95/PoseAug/f3f5c4e916ebf7529b873ec1c14c1ce0bf0f5cb1/models_baseline/gcn/sem_gcn.py'
    url_gcn_1 = 'https://raw.githubusercontent.com/jfzhang95/PoseAug/f3f5c4e916ebf7529b873ec1c14c1ce0bf0f5cb1/models_baseline/gcn/sem_graph_conv.py'

    gcn_out_0 = path.join(path_pth_5, 'sem_gcn.py')
    gcn_out_1 = path.join(path_pth_5, 'sem_graph_conv.py')

    print('downloading ' + path_pth_5 + '...')
    wget.download(url_gcn_0, gcn_out_0)
    wget.download(url_gcn_1, gcn_out_1)
    print('\ndone')

    oldString = 'from models_baseline.gcn.'
    newString = 'from '
    inplace_change(gcn_out_0, oldString, newString)


def get_blazepose_from_mediapipe():
    vvv = '9'
    url_full = 'https://github.com/google/mediapipe/raw/v0.8.' + vvv + '/mediapipe/modules/pose_landmark/pose_landmark_full.tflite'
    url_heavy = 'https://github.com/google/mediapipe/raw/v0.8.' + vvv + '/mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite'
    url_lite = 'https://github.com/google/mediapipe/raw/v0.8.' + vvv + '/mediapipe/modules/pose_landmark/pose_landmark_lite.tflite'

    path_tflite = 'methods/BLAZEPOSE/tflite_files'

    if not os.path.isdir(path_tflite):
        os.makedirs(path_tflite)

    VAR_full = 'pose_landmark_full_v08' + vvv + '.tflite'
    VAR_heavy = 'pose_landmark_heavy_v08' + vvv + '.tflite'
    VAR_lite = 'pose_landmark_lite_v08' + vvv + '.tflite'

    out_0 = path.join(path_tflite, VAR_full)
    out_1 = path.join(path_tflite, VAR_heavy)
    out_2 = path.join(path_tflite, VAR_lite)

    print('downloading ' + VAR_full + '...')
    wget.download(url_full, out_0)
    print('\ndone')
    print('downloading ' + VAR_heavy + '...')
    wget.download(url_heavy, out_1)
    print('\ndone')
    print('downloading ' + VAR_lite + '...')
    wget.download(url_lite, out_2)
    print('\ndone')


if __name__ == '__main__':
    get_yolox_from_mmdet()
    get_hrnet_from_mmpose()
    get_poseaug_from_authorsgit()
    get_blazepose_from_mediapipe()

    wally = 'here'

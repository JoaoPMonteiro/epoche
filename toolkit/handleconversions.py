#
import os
from os import path
import blobconverter
import torch
import onnxruntime
import numpy as np
import methodspaths


number_shaves = 4
opset_version = 11


def check_onnx_(ref_output, path_to_onnx, probe_img):
    c_session = onnxruntime.InferenceSession(path_to_onnx)
    c_input = {c_session.get_inputs()[0].name: probe_img}
    onnx_results = c_session.run(None, c_input)

    if isinstance(onnx_results, list):
        onnx_results = onnx_results[0]
    else:
        pass

    assert len(ref_output) == len(onnx_results)
    for pt_result, onnx_result in zip(ref_output, onnx_results):
        assert np.allclose(
            pt_result, onnx_result, atol=1.e-5
        ), 'The outputs are different between Pytorch and ONNX'
    print('The numerical values are same between Pytorch and ONNX')


def process_blazepose():
    vvv = '4'
    # ------------------------------------------------------------------------------------------------------
    # tflite to onnx
    import tf2onnx

    path_onnx = '../methods/BLAZEPOSE/onnx_files'

    if not os.path.isdir(path_onnx):
        os.makedirs(path_onnx)

    def export_tflite_to_onnx(in_m, out_m):
        tflite_graphs, opcodes_map, model, tensor_shapes = tf2onnx.tflite_utils.read_tflite_model(in_m)
        g = tf2onnx.tfonnx.process_tf_graph(None, tflite_path=in_m, shape_override=tensor_shapes)
        g = tf2onnx.optimizer.optimize_graph(g)
        model_proto = g.make_model('test')
        tf2onnx.utils.save_protobuf(out_m, model_proto)

    landmark_in_full = '../methods/BLAZEPOSE/tflite_files/pose_landmark_full_v08' + vvv + '.tflite'
    landmark_onnx_full = path.join(path_onnx, 'pose_landmark_full_v08' + vvv + '.onnx')
    landmark_in_heavy = '../methods/BLAZEPOSE/tflite_files/pose_landmark_heavy_v08' + vvv + '.tflite'
    landmark_onnx_heavy = path.join(path_onnx, 'pose_landmark_heavy_v08' + vvv + '.onnx')
    landmark_in_lite = '../methods/BLAZEPOSE/tflite_files/pose_landmark_lite_v08' + vvv + '.tflite'
    landmark_onnx_lite = path.join(path_onnx, 'pose_landmark_lite_v08' + vvv + '.onnx')

    export_tflite_to_onnx(landmark_in_full, landmark_onnx_full)
    export_tflite_to_onnx(landmark_in_heavy, landmark_onnx_heavy)
    export_tflite_to_onnx(landmark_in_lite, landmark_onnx_lite)

    # ------------------------------------------------------------------------------------------------------
    # onnx to myriadX blob

    def export_onnx_blob(in_m, out_m):

        blob_path = blobconverter.from_onnx(
            model=in_m,
            optimizer_params=[
                "--data_type=FP16",
                "--reverse_input_channels"
            ],
            compile_params=[
                "-ip fp16"
            ],
            output_dir=out_m,
            shaves=number_shaves,
        )

    path_blob = '../methods/BLAZEPOSE/blob_files'

    if not os.path.isdir(path_blob):
        os.makedirs(path_blob)

    export_onnx_blob(landmark_onnx_full, path_blob)
    export_onnx_blob(landmark_onnx_heavy, path_blob)
    export_onnx_blob(landmark_onnx_lite, path_blob)


def process_yolox():
    from mmdet.apis import init_detector
    import customsnippets

    # ------------------------------------------------------------------------------------------------------
    # pytorch to onnx

    path_onnx = '../methods/YOLOX/onnx_files'

    if not os.path.isdir(path_onnx):
        os.makedirs(path_onnx)

    path_blob = '../methods/YOLOX/blob_files'

    if not os.path.isdir(path_blob):
        os.makedirs(path_blob)

    in_chckpnt = '../methods/YOLOX/pth_files/yolox_tiny_8x8_300e_coco.pth'
    in_cnfg = '../methods/YOLOX/config_files/yolox_tiny_8x8_300e_coco.py'
    output_file = '../methods/YOLOX/onnx_files/yolox_tiny_8x8_300e_coco.onnx'

    model = init_detector(in_cnfg, in_chckpnt, device='cpu')

    with torch.no_grad():

        custom_yolox = customsnippets.build_custom_yolox_from_model(model)
        custom_yolox.forward = custom_yolox.customforward
        custom_yolox.cpu().eval()
        one_img = torch.randn([1, 3, 640, 640], requires_grad=False)
        pytorch_results = custom_yolox(one_img)

        torch.onnx.export(
            custom_yolox,
            one_img,
            output_file,
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=True,
            opset_version=opset_version)

    check_onnx_(pytorch_results.detach().cpu(), output_file, one_img.detach().cpu().numpy())

    # ------------------------------------------------------------------------------------------------------
    # onnx to myriadX blob

    blob_path = blobconverter.from_onnx(
        model=output_file,
        optimizer_params=[
            "--data_type=FP16",
        ],
        compile_params=[
            "-ip U8 -op fp16"
        ],
        output_dir=path_blob,
        shaves=number_shaves,
    )


def process_hrnet():
    from mmpose.apis import init_pose_model

    # ------------------------------------------------------------------------------------------------------
    # pytorch to onnx

    path_onnx = '../methods/HRNET/onnx_files'

    if not os.path.isdir(path_onnx):
        os.makedirs(path_onnx)

    path_blob = '../methods/HRNET/blob_files'

    if not os.path.isdir(path_blob):
        os.makedirs(path_blob)

    in_chckpnt = '../methods/HRNET/pth_files/hrnet_w32_mpii_256x256.pth'
    in_cnfg = '../methods/HRNET/config_files/hrnet_w32_mpii_256x256.py'
    output_file = '../methods/HRNET/onnx_files/hrnet_w32_mpii_256x256.onnx'

    model = init_pose_model(in_cnfg, in_chckpnt, device='cpu')
    model.forward = model.forward_dummy

    model.cpu().eval()

    one_img = torch.randn([1, 3, 256, 256], requires_grad=False)
    pytorch_results = model(one_img)

    torch.onnx.export(
        model,
        one_img,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=False,
        verbose=True,
        opset_version=opset_version)

    check_onnx_(pytorch_results.detach().cpu(), output_file, one_img.detach().cpu().numpy())

    # ------------------------------------------------------------------------------------------------------
    # onnx to myriadX blob

    blob_path = blobconverter.from_onnx(
        model=output_file,
        optimizer_params=[
            "--data_type=FP16",
        ],
        compile_params=[
            "-ip fp16 -op fp16"
        ],
        output_dir=path_blob,
        shaves=number_shaves,
    )


def process_method(in_method, optional=None):
    if optional is None:
        x_c = in_method.upper()
        path_onnx = '../methods/' + x_c + '/onnx_files'
        if not os.path.isdir(path_onnx):
            os.makedirs(path_onnx)
        path_blob = '../methods/' + x_c + '/blob_files'
        if not os.path.isdir(path_blob):
            os.makedirs(path_blob)
    else:
        x_c = in_method.upper()
        s_c = optional.upper()
        path_onnx = '../methods/' + x_c + '/' + s_c + '/onnx_files'
        if not os.path.isdir(path_onnx):
            os.makedirs(path_onnx)
        path_blob = '../methods/' + x_c + '/' + s_c + '/blob_files'
        if not os.path.isdir(path_blob):
            os.makedirs(path_blob)

        in_method = optional + in_method

    import handleestimation
    with torch.no_grad():
        p_l = handleestimation.PoseLifting(method=in_method, architecture='pytorch')
        model_pos = p_l.get_model()
        model_pos.eval()

        one_img = torch.randn([1, 16, 2], requires_grad=False)
        pytorch_results = model_pos(one_img)

        output_file = methodspaths.methodsDict[in_method+'_Paths'].onnx

        torch.onnx.export(model_pos,
                          one_img,
                          output_file,
                          opset_version=12,
                          export_params=True,
                          keep_initializers_as_inputs=False,
                          verbose=True,
                          input_names=['input'],
                          output_names=['output'])

    check_onnx_(pytorch_results.detach().cpu(), output_file, one_img.detach().cpu().numpy())

    # ------------------------------------------------------------------------------------------------------
    # onnx to myriadX blob

    blob_path = blobconverter.from_onnx(
        model=output_file,
        optimizer_params=[
            "--data_type=FP16",
        ],
        compile_params=[
            "-ip fp16 -op fp16"
        ],
        output_dir=path_blob,
        shaves=number_shaves,
    )


def process_poseaug():
    process_method('vpose')
    process_method('vpose', optional='poseaug')
    process_method('sbl')
    process_method('sbl', optional='poseaug')
    process_method('gcn')
    process_method('gcn', optional='poseaug')
    process_method('stgcn')
    process_method('stgcn', optional='poseaug')


if __name__ == '__main__':
    process_blazepose()
    process_yolox()
    process_hrnet()
    process_poseaug()

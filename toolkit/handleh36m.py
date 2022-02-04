from os import path
from tqdm import tqdm
import h5py
import cv2

# ///////////////////////////////////////////////////////////////////////////////////////////
# (adapted from) https://github.com/anibali/h36m-fetch
import xml.etree.ElementTree as ET


class H36M_Metadata:
    def __init__(self, metadata_file):
        self.subjects = []
        self.sequence_mappings = {}
        self.action_names = {}
        self.camera_ids = []

        tree = ET.parse(metadata_file)
        root = tree.getroot()

        for i, tr in enumerate(root.find('mapping')):
            if i == 0:
                _, _, *self.subjects = [td.text for td in tr]
                self.sequence_mappings = {subject: {} for subject in self.subjects}
            elif i < 33:
                action_id, subaction_id, *prefixes = [td.text for td in tr]
                for subject, prefix in zip(self.subjects, prefixes):
                    self.sequence_mappings[subject][(action_id, subaction_id)] = prefix

        for i, elem in enumerate(root.find('actionnames')):
            action_id = str(i + 1)
            self.action_names[action_id] = elem.text

        self.camera_ids = [elem.text for elem in root.find('dbcameras/index2id')]

    def get_base_filename(self, subject, action, subaction, camera):
        return '{}.{}'.format(self.sequence_mappings[subject][(action, subaction)], camera)


def load_h36m_metadata(src_dir):
    path_to_metadata_file = path.join(src_dir, 'metadata.xml')
    return H36M_Metadata(path_to_metadata_file)


# ///////////////////////////////////////////////////////////////////////////////////////////


def preprocess3d_h36m(in_h36m_sk):
    '''
    :param in_3dhp_sk: input h36m 3d skeleton annotation
    :return: 16 point skeleton
    '''
    selected_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
    ou_h36m_sk = in_h36m_sk[selected_joints]

    return ou_h36m_sk


def preprocess2d_h36m(in_h36m_sk):
    '''
    :param in_h36m_sk: input h36m 2d skeleton annotation
    :return: 16 point skeleton
    '''
    selected_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
    ou_h36m_sk = in_h36m_sk[selected_joints]

    return ou_h36m_sk


def explore_test_data(src_dir):
    h36m_metadata = load_h36m_metadata(src_dir)
    sequence_mappings = h36m_metadata.sequence_mappings
    test_subjects = {
        'S9': 9,
        'S11': 11
    }

    subactions = []
    for subject in test_subjects.keys():
        subactions += [
            (subject, action, subaction)
            for action, subaction in sequence_mappings[subject].keys()
            if int(action) > 1  # Exclude '_ALL'
        ]

    for subject, action, subaction in tqdm(subactions, ascii=True, leave=False):
        out_dir_r = path.join('processed', subject, h36m_metadata.action_names[action] + '-' + subaction)
        out_dir = path.join(src_dir, out_dir_r)
        f = h5py.File(path.join(out_dir, 'annot.h5'), 'r')
        poses_3d = f['pose']['3d']
        poses_3d_univ = f['pose']['3d-univ']
        poses_2d = f['pose']['2d']
        frames = f['frame']
        cam_id = f['camera']

        for i, j in enumerate(frames):
            c_cam = str(cam_id[i])
            filename = 'img_%06d.jpg' % j
            ref_path = path.join(out_dir, 'imageSequence')
            cam_path = path.join(ref_path, c_cam)
            img_path = path.join(cam_path, filename)

            image = cv2.imread(img_path)
            annot2 = poses_2d[i]
            annot3 = poses_3d[i]
            univ_annot3 = poses_3d_univ[i]

            p_annot2 = preprocess2d_h36m(annot2)
            p_annot3 = preprocess3d_h36m(annot3)

            import assortedroutines
            img_w = image.shape[0]
            img_h = image.shape[1]
            assortedroutines.another_2d_plot(p_annot2, img_w, img_h)
            assortedroutines.another_3d_plot(p_annot3)

            wally = 'here'


# ----------------------------------------------------------------------------------------------
def test():
    import configparser

    config = configparser.ConfigParser()
    config.read('../config.ini')
    path_to_h36m_folder = config['General']['PATHTOH36MFOLDER']

    explore_test_data(path_to_h36m_folder)
    wally = 'here'


# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    test()

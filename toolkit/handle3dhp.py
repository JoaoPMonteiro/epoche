import cv2

# ///////////////////////////////////////////////////////////////////////////////////////////
# (adapted from) https://github.com/anibali/margipose/
from tqdm import tqdm
import h5py
from os import path
import numpy as np


def _progress(iterator, name):
    return tqdm(iterator, desc='{:10s}'.format(name), ascii=True, leave=False)


class RawMpiTestSeqDataset:
    def __init__(self, data_dir, seq_id, valid_only=True):
        frame_indices = []

        with h5py.File(path.join(data_dir, seq_id, 'annot_data.mat'), 'r') as annot:
            if valid_only:
                new_frame_indices = list(np.where(annot['valid_frame'])[0])
            else:
                new_frame_indices = list(range(len(annot['valid_frame'])))
        frame_indices += new_frame_indices

        self.data_dir = data_dir
        self.frame_indices = frame_indices
        self.seq_id = seq_id
        self.annot_file = path.join(self.data_dir, self.seq_id, 'annot_data.mat')

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, index):
        frame_index = self.frame_indices[index]
        image_id = frame_index + 1
        image_file = path.join(
            self.data_dir, self.seq_id, 'imageSequence', 'img_%06d.jpg' % image_id)

        with h5py.File(self.annot_file, 'r') as annot:
            return {
                'image_file': image_file,
                'seq_id': self.seq_id,
                'frame_index': frame_index,
                'valid': int(annot['valid_frame'][frame_index][0]),
                'annot2': annot['annot2'][frame_index][0],
                'annot3': annot['annot3'][frame_index][0],
                'univ_annot3': annot['univ_annot3'][frame_index][0]
            }


class RawMpiTestDataset:
    # Names of test sequences
    SEQ_IDS = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']

    def __init__(self, data_dir, valid_only=True):
        self.seq_datasets = [
            RawMpiTestSeqDataset(data_dir, seq_id, valid_only=valid_only)
            for seq_id in self.SEQ_IDS
        ]

        seq_indices = []
        frame_indices = []
        seq_start_indices = {}
        for seq_index, seq_dataset in enumerate(self.seq_datasets):
            seq_start_indices[seq_dataset.seq_id] = len(frame_indices)
            frame_indices += list(range(len(seq_dataset)))
            seq_indices += [seq_index] * len(seq_dataset)

        self.data_dir = data_dir
        self.frame_indices = frame_indices
        self.seq_indices = seq_indices
        self.seq_start_indices = seq_start_indices

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, index):
        return self.seq_datasets[self.seq_indices[index]][self.frame_indices[index]]

# ///////////////////////////////////////////////////////////////////////////////////////////


def preprocess3d_3dhp(in_3dhp_sk):
    '''
    :param in_3dhp_sk: input 3dhp 3d skeleton annotation
    :return: 16 point skeleton
    '''
    selected_joints = [14, 8, 9, 10, 11, 12, 13, 15, 1, 0, 5, 6, 7, 2, 3, 4]
    ou_3dhp_sk = in_3dhp_sk[selected_joints]

    return ou_3dhp_sk


def preprocess2d_3dhp(in_3dhp_sk):
    '''
    :param in_3dhp_sk: input 3dhp 2d skeleton annotation
    :return: 16 point skeleton
    '''
    selected_joints = [14, 8, 9, 10, 11, 12, 13, 15, 1, 0, 5, 6, 7, 2, 3, 4]
    ou_3dhp_sk = in_3dhp_sk[selected_joints]

    return ou_3dhp_sk


def explore_test_data(src_dir):

    for seq_id in _progress(RawMpiTestDataset.SEQ_IDS, 'Sequences'):
        dataset = RawMpiTestSeqDataset(src_dir, seq_id, valid_only=True)

        for example in _progress(dataset, 'Images'):
            image = cv2.imread(example['image_file'])
            annot2 = example['annot2']
            annot3 = example['annot3']
            univ_annot3 = example['univ_annot3']

            p_annot2 = preprocess2d_3dhp(annot2)
            p_annot3 = preprocess3d_3dhp(annot3)

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
    path_to_3dhp_folder = config['General']['PATHTO3DHPFOLDER']

    explore_test_data(path_to_3dhp_folder)
    wally = 'here'


# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    test()

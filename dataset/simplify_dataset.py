from tqdm import tqdm
import numpy as np
import os
import scipy.io as scio


def simplify_grasp_labels(root, save_path):
    """
    original dataset grasp_label files have redundant data,  We can significantly save the memory cost
    """
    obj_names = list(range(88))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in obj_names:
        print('\nsimplifying object {}:'.format(i))
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        point_num = len(label['points'])
        print('original shape:               ', label['points'].shape, label['offsets'].shape, label['scores'].shape)
        if point_num > 4820:
            idxs = np.random.choice(point_num, 4820, False)
            points = label['points'][idxs]
            offsets = label['offsets'][idxs]
            scores = label['scores'][idxs]
            print('Warning!!!  down sample object {}'.format(i))
        else:
            points = label['points']
            scores = label['scores']
            offsets = label['offsets']
        width = offsets[:, :, :, :, 2]
        print('after simplify, offset shape: ', points.shape, scores.shape, width.shape)
        np.savez(os.path.join(save_path, '{}_labels.npz'.format(str(i).zfill(3))),
                 points=points, scores=scores, width=width)


if __name__ == '__main__':
    root = '/media/bot/980A6F5E0A6F38801/datasets/graspnet/'
    save_path = os.path.join(root, 'grasp_label_simplified')
    simplify_grasp_labels(root, save_path)
    # a = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(0).zfill(3))))
    # b = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(0).zfill(3))))
    # print(a['offsets'], b['width'], a['points'], b['points'],  a['scores'], b['scores'])
    # print(np.sum(a['offsets'][:, :, :, :, 2] - b['width']), np.max(b['width']), np.max(a['offsets'][:, :, :, :, 2]))

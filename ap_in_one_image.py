import numpy as np
from graspnetAPI import GraspGroup
import os
import pickle
import open3d as o3d
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.eval_utils import create_table_points, parse_posevector, transform_points, \
    eval_grasp, load_dexnet_model, voxel_sample_points
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def get_config_rgbmatter():
    '''
     - return the config dict
    '''
    config = dict()
    config['scene_id'] = 100
    config['index_str'] = '0000'
    config['camera'] = 'kinect'
    config['collision_detection_choice'] = 'point_cloud'  # point_cloud or full_model

    # config['dataset_path'] = "/data3/graspnet/"
    config['dataset_path'] = '/media/bot/980A6F5E0A6F38801/datasets/graspnet'

    # config['res_6D_pose_path'] = '/home/zibo/cvpr2022/graspnet-cvpr/logs/' \
    #                              'log_kn_graspness_minkowski_v0/dump_epoch04_fps'
    config['res_6D_pose_path'] = '/data/zibo/logs/log_kn_v1/dump_epoch06_fps/'
    config['scene_id_str'] = 'scene_' + str(config['scene_id']).zfill(4)
    return config


my_config = get_config_rgbmatter()

res_6d_path = os.path.join(my_config['res_6D_pose_path'], my_config['scene_id_str'], my_config['camera'],
                           my_config['index_str'] + '.npy')
# res_6d_path = os.path.join(my_config['grasp_gt_path'], my_config['scene_id_str'], my_config['camera'],
#                            my_config['index_str'] + '.npy')
# res_6d_path = my_config['res_6D_pose_path']
grasp = np.load(res_6d_path)
print('\ntesting {}, scene_{} image_{}'.format(res_6d_path, my_config['scene_id'], my_config['index_str']))

grasp_group = GraspGroup(grasp)
print('grasp shape: ', grasp_group.grasp_group_array.shape)
camera_pose = np.load(os.path.join(my_config['dataset_path'], 'scenes', my_config['scene_id_str'], my_config['camera'],
                                   'camera_poses.npy'))[int(my_config['index_str'])]
align_mat = np.load(os.path.join(my_config['dataset_path'], 'scenes', my_config['scene_id_str'],
                                 my_config['camera'], 'cam0_wrt_table.npy'))
scene_reader = xmlReader(os.path.join(my_config['dataset_path'], 'scenes', my_config['scene_id_str'],
                                      my_config['camera'], 'annotations', my_config['index_str'] + '.xml'))

config = get_config()
TOP_K = 50
list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
max_width = 0.1
model_dir = os.path.join(my_config['dataset_path'], 'models')

posevectors = scene_reader.getposevectorlist()
obj_list = []
pose_list = []
for posevector in posevectors:
    obj_idx, mat = parse_posevector(posevector)
    obj_list.append(obj_idx)
    pose_list.append(mat)
table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

gg_array = grasp_group.grasp_group_array
min_width_mask = (gg_array[:, 1] < 0)
max_width_mask = (gg_array[:, 1] > max_width)
gg_array[min_width_mask, 1] = 0
gg_array[max_width_mask, 1] = max_width
grasp_group.grasp_group_array = gg_array

obj_list = []
model_list = []
dexmodel_list = []
for posevector in posevectors:
    obj_idx, _ = parse_posevector(posevector)
    obj_list.append(obj_idx)
for obj_idx in obj_list:
    model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
    dex_cache_path = os.path.join(my_config['dataset_path'], "dex_models", '%03d.pkl' % obj_idx)
    if os.path.exists(dex_cache_path):   # don't know why in 203, pickle.load() will throw an error, so add not
        with open(dex_cache_path, 'rb') as f:
            dexmodel = pickle.load(f)
    else:
        dexmodel = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_idx, 'textured'))
    points = np.array(model.points)
    model_list.append(points)
    dexmodel_list.append(dexmodel)

model_sampled_list = list()
for model in model_list:
    model_sampled = voxel_sample_points(model, 0.008)
    model_sampled_list.append(model_sampled)

grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list, pose_list,
                                                         config, table=table_trans, voxel_size=0.008, TOP_K=TOP_K)
# remove empty
grasp_list = [x for x in grasp_list if len(x) != 0]
score_list = [x for x in score_list if len(x) != 0]
collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

if len(grasp_list) == 0:
    grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
    print('\rMean Accuracy {}'.format(np.mean(grasp_accuracy[:, :])))

# concat into scene level
grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(
    score_list), np.concatenate(collision_mask_list)
# sort in scene level
print('final grasp shape that used to compute ap: ', grasp_list.shape)
grasp_confidence = grasp_list[:, 0]
indices = np.argsort(-grasp_confidence)
grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[
    indices]

# calculate AP
grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
print(score_list)
# score_list = np.array([i for i in score_list if abs(i + 3) > 1.001])
for fric_idx, fric in enumerate(list_coe_of_friction):
    for k in range(0, TOP_K):
        if k + 1 > len(score_list):
            grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (
                    k + 1)
        else:
            grasp_accuracy[k, fric_idx] = np.sum(
                ((score_list[0:k + 1] <= fric) & (score_list[0:k + 1] > 0)).astype(int)) / (k + 1)
# print(grasp_accuracy)
print('\nMean Accuracy: %.3f' % (100.0 * np.mean(grasp_accuracy[:, :])))

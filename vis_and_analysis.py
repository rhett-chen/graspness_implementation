import numpy as np
from graspnetAPI import GraspGroup
import open3d as o3d
import scipy.io as scio
from PIL import Image
import os
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask

data_path = '/media/bot/980A6F5E0A6F38801/datasets/graspnet'
dump_dir = '/data/zibo/logs/log_kn_v1/dump_epoch05_fps/'
scene_id = 'scene_0101'
ann_id = '0000'

poses = np.load(os.path.join(dump_dir, scene_id, 'kinect', ann_id + '.npy'))

#  vis
color = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, 'kinect', 'rgb', ann_id + '.png')), dtype=np.float32) / 255.0
depth = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, 'kinect', 'depth', ann_id + '.png')))
seg = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, 'kinect', 'label', ann_id + '.png')))
meta = scio.loadmat(os.path.join(data_path, 'scenes', scene_id, 'kinect', 'meta', ann_id + '.mat'))
intrinsic = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']
camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
point_cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
depth_mask = (depth > 0)
camera_poses = np.load(os.path.join(data_path, 'scenes', scene_id, 'kinect', 'camera_poses.npy'))
align_mat = np.load(os.path.join(data_path, 'scenes', scene_id, 'kinect', 'cam0_wrt_table.npy'))
trans = np.dot(align_mat, camera_poses[int(ann_id)])
workspace_mask = get_workspace_mask(point_cloud, seg, trans=trans, organized=True, outlier=0.02)
mask = (depth_mask & workspace_mask)
point_cloud = point_cloud[mask]
color = color[mask]

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3).astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3).astype(np.float32))
gg = GraspGroup(poses)
gg = gg.nms()
# print(gg.grasp_group_array.shape)
gg = gg.sort_by_score()
if gg.__len__() > 30:
    gg = gg[:30]
grippers = gg.to_open3d_geometry_list()
o3d.visualization.draw_geometries([cloud, *grippers])

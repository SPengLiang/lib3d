import os
import numpy as np
import cv2 as cv
from tqdm import tqdm


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def corner_3d(bbox3d_input):
    '''

    Args:
        bbox3d_input: a single bbox list with coordinates. format:(6)  [h, w, l, x, y, z, Ry]

    Returns:
        8 corners of the 3d box. format: narray(8*3) [h, w, l, x, y, z, Ry]
    '''
    '''
        vertex define

       0  ____ 1 
      4 /|___/|            / z(l)
       / /  / / 5         /
    3 /_/_2/ /           /----->x(w)
      |/___|/            |
    7     6              | y(h)
    '''
    bbox3d = bbox3d_input.copy()

    R = roty(bbox3d[-1])

    # # 3d bounding box dimensions
    h = bbox3d[0]
    w = bbox3d[1]
    l = bbox3d[2]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d_shift = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d = np.transpose(corners_3d_shift) + np.array([bbox3d[3], bbox3d[4], bbox3d[5]])

    return corners_3d


def convert_to_3d(depth, P2, upsample_factor, x_start, y_start):
    '''
    :param depth: depth map of current frame cropped area. SHAPE: A*B
    :param P2: projection matrix of left RGB camera.  SHAPE: 4*3
    :param upsample_factor: upsample factor of the cropped area.
    :param x_start: start coordinates in image coordinates of x.
    :param y_start: start coordinates in image coordinates of y.
    :return:
            points: 3D coordinates in real world of cropped area.   SHAPE: N*3
            uv_points: corresponding 2D coordinates in image coordinates of 3D points  SHAPE: N*2
    '''
    fx = P2[0][0] * upsample_factor
    fy = P2[1][1] * upsample_factor
    cx = P2[0][2] * upsample_factor
    cy = P2[1][2] * upsample_factor
    # print(fx, fy, cx, cy)

    x_tile = np.array(range(depth.shape[1])).reshape(1, -1) + x_start
    points_x = np.tile(x_tile, [depth.shape[0], 1])

    y_tile = np.array(range(depth.shape[0])).reshape(-1, 1) + y_start
    points_y = np.tile(y_tile, [1, depth.shape[1]])

    points_x = points_x.reshape((-1, 1))
    points_y = points_y.reshape((-1, 1))
    depth = depth.reshape((-1, 1))

    # -------mask-------
    # mask = np.where(depth != np.inf, True, False)
    # points_x = points_x[mask].reshape((-1, 1))
    # points_y = points_y[mask].reshape((-1, 1))
    # depth = depth[mask].reshape((-1, 1))

    uv_points = np.concatenate([points_x, points_y], axis=1)

    points_x = (points_x - cx) / fx
    points_y = (points_y - cy) / fy

    points_x *= depth
    points_y *= depth

    points = np.concatenate([points_x, points_y, depth], axis=1)

    return points, uv_points

def convert_to_2d(point_cloud, P2):
    '''
    :param point_cloud: 3D point_cloud. SHAPE: N*3
    :param P2: projection matrix of left RGB camera.  SHAPE: 4*3
    :return:
            points_2d: 3D coordinates project to image plane.  SHAPE: N*3 [u, v, depth]
    '''
    P2_inv_coor = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
    P2_inv = np.linalg.inv(P2_inv_coor)

    point_cloud_coor = np.concatenate([point_cloud, np.ones((point_cloud.shape[0], 1))], axis=1)
    points_2d = np.dot(P2_inv, point_cloud_coor.T).T

    return points_2d

def convert_kitti_lidar_to_depth(kitti_calib_path, kitti_lidar_path, img_h, img_w):
    '''
    Args:
        kitti_calib_path: current kitti image calib file path
        kitti_lidar_path: current kitti lidar file path
        img_h: current image height
        img_w: current image width

    Returns:
        depth: lidar convert to depth image
    '''

    lidar_data = np.fromfile(str(kitti_lidar_path), dtype=np.float32, count=-1).reshape([-1,4])
    lidar_data = np.concatenate([lidar_data[:, :3], np.ones((lidar_data.shape[0], 1))], axis=1)

    calib = parse_kitti_calib(kitti_calib_path)
    P2 = calib['P2']
    R0_rect = np.eye(4, dtype= np.float32)
    Tr_velo_to_cam = np.eye(4, dtype= np.float32)

    R0_rect[:3, :3] = calib['R0_rect']
    Tr_velo_to_cam[:3, :4] = calib['Tr_velo_to_cam']

    trans_mat = np.dot(np.dot(P2, R0_rect), Tr_velo_to_cam)
    trans_2d = np.dot(trans_mat, lidar_data.T).T
    trans_2d[:, [0, 1]] /= trans_2d[:, [2]]

    trans_2d = trans_2d[(trans_2d[:, 0] > 0) & (trans_2d[:, 0] < img_w) &
                        (trans_2d[:, 1] > 0) & (trans_2d[:, 1] < img_h) &
                        (trans_2d[:, 2] > 0)]

    depth = np.zeros((img_h, img_w))
    depth[trans_2d[:, 1].astype(np.int32), trans_2d[:, 0].astype(np.int32)] = trans_2d[:, 2]

    return depth


def rot_pc(f_angle, pc):
    '''
    Args:
        f_angle: rotated angle (axis y)
        pc: 3D coordinates in real world of cropped area.  SHAPE: N*3

    Returns:
        rotated 3d points.

    '''
    R = roty(f_angle)
    r_p = np.dot(R, pc.T).T

    return r_p


def parse_kitti_calib(calib_path):
    with open(calib_path, encoding='utf-8') as f:
        text = f.readlines()
        P0 = np.array(text[0].split(' ')[1:], dtype=np.float32).reshape(3, 4)
        P1 = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 4)
        P2 = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)
        P3 = np.array(text[3].split(' ')[1:], dtype=np.float32).reshape(3, 4)

        R0_rect = np.array(text[4].split(' ')[1:], dtype=np.float32).reshape(3, 3)
        Tr_velo_to_cam = np.array(text[5].split(' ')[1:], dtype=np.float32).reshape(3, 4)
        Tr_imu_to_velo = np.array(text[6].split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib = {
            'P0': P0,
            'P1': P1,
            'P2': P2,
            'P3': P3,
            'R0_rect': R0_rect,
            'Tr_velo_to_cam': Tr_velo_to_cam,
            'Tr_imu_to_velo': Tr_imu_to_velo,
        }
    return calib

def convert_color_map(img):
    temp = img / np.max(img) * 255
    temp = temp.astype(np.uint8)
    im_color = cv.applyColorMap(temp, cv.COLORMAP_JET)
    return im_color


# img = cv.imread('/private/pengliang/VNL/depth/000000.png', -1) / 256.
# img = 384 / img
# cv.imwrite('d3.png', convert_color_map(img))

if __name__ == '__main__':
    lidar_path = '/private/pengliang/KITTI3D/training/velodyne/'
    calib_path = '/private/pengliang/KITTI3D/training/calib/'
    left_path = '/private/pengliang/KITTI3D/training/image_2/'
    lidar_depth_path = '/private/pengliang/KITTI3D/training/lidar_depth/'

    all_files = sorted(os.listdir(lidar_path))

    for f in tqdm(all_files):
        img = cv.imread(os.path.join(left_path, f[:-4]+'.png'))
        depth_img = convert_kitti_lidar_to_depth(calib_path+f[:-4]+'.txt',
                                                 lidar_path+f,
                                                 img.shape[0],
                                                 img.shape[1])
        cv.imwrite(os.path.join(lidar_depth_path, f[:-4]+'.png'), (depth_img*256).astype(np.uint16))

    # lidar_path = '/private/pengliang/KITTI3D/training/velodyne/000003.bin'
    # calib_path = '/private/pengliang/KITTI3D/training/calib/000003.txt'
    #
    # lidar_data = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1,4])
    # lidar_data = np.concatenate([lidar_data[:, :3], np.ones((lidar_data.shape[0], 1))], axis=1)
    #
    # calib = parse_kitti_calib(calib_path)
    # P2 = calib['P2']
    # R0_rect = np.eye(4, dtype= np.float32)
    # Tr_velo_to_cam = np.eye(4, dtype= np.float32)
    #
    # R0_rect[:3, :3] = calib['R0_rect']
    # Tr_velo_to_cam[:3, :4] = calib['Tr_velo_to_cam']
    #
    # trans_mat = np.dot(np.dot(P2, R0_rect), Tr_velo_to_cam)
    # trans_2d = np.dot(trans_mat, lidar_data.T).T
    # trans_2d[:, [0, 1]] /= trans_2d[:, [2]]
    #
    # trans_2d = trans_2d[(trans_2d[:, 0] > 0) & (trans_2d[:, 0] < 1275) &
    #                     (trans_2d[:, 1] > 0) & (trans_2d[:, 1] < 374)]
    #
    # img = np.zeros((374, 1275))
    # img[trans_2d[:, 1].astype(np.int32), trans_2d[:, 0].astype(np.int32)] = trans_2d[:, 2]
    # disp = np.zeros_like(img)
    # disp[img > 0] = 384 / img[img > 0]
    # cv.imwrite('d3.png', convert_color_map(disp))

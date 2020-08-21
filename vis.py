import numpy as np
import cv2 as cv

class Vis3d():
    def __init__(self, range_lr, range_fb, resolution):
        self.l_b = range_lr[0]
        self.r_b = range_lr[1]
        self.b_b = range_fb[0]
        self.f_b = range_fb[1]
        self.resolution = resolution
        self.reset()

    def reset(self):
        self.map = np.zeros((int((self.f_b-self.b_b)*self.resolution),
                             int((self.r_b-self.l_b)*self.resolution),
                             3))

    def get_map(self):
        return self.map

    def _roty2d(self, t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, s],
                         [-s, c]])

    def add_bev_box(self, bev_center, wl, ry, color, thk=5):
        w, l = wl
        corners = np.array([[-w/2, -w/2, w/2, w/2],
                            [-l/2, l/2, l/2, -l/2]])
        R = self._roty2d(-ry)
        rot_corners = np.dot(R, corners).T + bev_center
        img_corners = self._convert_real2img(rot_corners)
        for i in range(4):
            self._draw_line(img_corners[i], img_corners[(i+1)%4], color, thk)

    def _convert_real2img(self, points):
        img_x = (points[:, 0] - self.l_b) * self.resolution
        img_y = (self.f_b-self.b_b)*self.resolution - \
                (points[:, 1] - self.b_b) * self.resolution
        return np.stack([img_x.astype(np.uint16), img_y.astype(np.uint16)], axis=1)

    def _draw_line(self, p1, p2, color, thk):
        self.map = cv.line(self.map, (p1[0], p1[1]), (p2[0], p2[1]), color, thk)

    def _draw_circle(self, p1, color, thk=1):
        self.map = cv.circle(self.map, (p1[0], p1[1]), color, thk)

    def _draw_points(self, points, color):
        self.map[points[:, 1], points[:, 0]] = color
import numpy as np
import open3d as o3d


class Board:
    def __init__(self):
        self.square_length = 0.0395
        self.marker_length = 0.0315
        self.grid_num = 10

        # x+ right, y+ up
        # (0.004, -0.004, 0)---------- (0.036, -0.004, 0)
        #          |                           |
        #          |                           |
        #          |                           |
        #          |                           |
        # (0.004, -0.036, 0)---------- (0.036, -0.036, 0)
        # [[0.004, -0.004, 0], [0.036, -0.004, 0], [0.036, -0.036, 0], [0.004, -0.036, 0]]
        self.aruco_grid = np.array([
            [0, 0, 0],
            [self.marker_length, 0, 0],
            [self.marker_length, self.marker_length, 0],
            [0, self.marker_length, 0]
        ]) + np.array([
            [(self.square_length - self.marker_length) / 2, (self.square_length - self.marker_length) / 2, 0]
        ])

        x, y = np.meshgrid(np.arange(self.grid_num), np.arange(self.grid_num))
        self.grids = np.concatenate([x.reshape((self.grid_num, self.grid_num, 1)), y.reshape((self.grid_num, self.grid_num, 1))], axis=2).reshape(-1, 2)

        self.aruco_grids = []
        for grid in self.grids:
            if (grid[0] + grid[1]) % 2 == 0:
                temp = (np.array([grid[0], grid[1], 0]) * self.square_length + self.aruco_grid) * np.array([1, -1, 1])
                self.aruco_grids.append(temp)
        self.ids = range(len(self.aruco_grids))
        self.aruco_grids = np.array(self.aruco_grids)

        self.cloud_points = o3d.geometry.PointCloud()
        self.cloud_points.points = o3d.utility.Vector3dVector(self.aruco_grids.reshape(-1, 3))
        self.cloud_points.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]] * self.aruco_grids.reshape(-1, 3).shape[0]).tolist())

    def __getitem__(self, aruco_id: int):
        """
        :param aruco_id: int
        :return: shape: (N, 3)
            [[x1, y1, z1],
             [x2, y2, z2],
             [x3, y3, z3],
             [x4, y4, z4]      ]
        """
        if aruco_id < 0 or aruco_id >= len(self.aruco_grids):
            return None, aruco_id
        return self.aruco_grids[aruco_id], None

    def get_grids(self, mask):
        """

        :param mask: [2,4,5,6,8...] valid aruco ids
        :return: shape: (N, 4, 3)
        """
        grids = []
        for id in mask:
            if self.__getitem__(id)[0] is None:
                return None, id
            grids.append(self.__getitem__(id)[0])
        return np.array(grids).reshape([-1, 4, 3]), None


class Apriltag:
    def __init__(self):
        self.square_length = 0.042
        # x+ right, y+ up
        self.grids = np.array([
            [-1, 1, 0],
            [1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0]
        ]) * self.square_length / 2

        self.grids = self.grids[[3, 2, 1, 0]]

    def get_cloud_points(self, R_matrix, t_vector):
        cloud_points = o3d.geometry.PointCloud()
        pos = (R_matrix @ self.grids.T + t_vector.reshape(3, 1)).T
        print(pos.shape)
        cloud_points.points = o3d.utility.Vector3dVector(pos.tolist())
        cloud_points.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * 4).tolist())
        return cloud_points


if __name__ == '__main__':
    board = Board()
    # print(board)

import json
import time

import numpy as np
from typing import *
import open3d as o3d
import transforms3d as t3d
import math


class HandBoard:
    def __init__(self):
        self.mark_length = 0.06
        self.hand2mark_centric = 0.06
        # x+ right, y+ up
        # 4------1                               1------2
        # |      |-----(hand_id0) (hand_id1)-----|      |
        # 3------2                               4------3
        self.raw_grid_0 = np.array([
            [1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [-1, 1, 0],
        ]) * self.mark_length / 2 + np.array([-self.hand2mark_centric, 0, 0])
        self.raw_grid_0 = np.concatenate([self.raw_grid_0, np.zeros([1, 3])], axis=0)

        self.raw_grid_1 = -self.raw_grid_0[[1, 2, 3, 0], :]

        theta = math.pi / 2
        R_z = np.array([[math.cos(theta), -math.sin(theta), 0],
                        [math.sin(theta), math.cos(theta), 0],
                        [0, 0, 1]
                        ])

        self.raw_grid_0 = (R_z @ self.raw_grid_0.T).T

        self.map = {"LHR-656F5409": self.raw_grid_0, "LHR-6A736404": self.raw_grid_1}
        self.map2 = {0: "LHR-656F5409", 1: "LHR-6A736404"}

    def plot(self, R_matrix, T_vector, aruco_id, ax):
        grid = self.map[self.map2[aruco_id]].T
        print(grid.shape)
        grid = np.matmul(R_matrix, grid).reshape([3, -1]) + T_vector.reshape([3, -1])
        ax.scatter(grid[0], grid[1], grid[2], 'o')


class HandAruco:
    def __init__(self, name: str, exchange=None):
        if exchange is None:
            exchange = [0, 1, 2, 3]
        self.name: str = name
        self.mark_length = 0.055
        self.hand2mark_centric = 0.074
        # z- right, x- up
        #  3-----4
        #  |     |
        #  2-----1
        #     |
        # (hand_id0)
        #
        raw_grid = np.array([
            [1, 0, -1],
            [1, 0, 1],
            [-1, 0, 1],
            [-1, 0, -1],
        ]) * self.mark_length / 2 + np.array([-self.hand2mark_centric, 0, 0])

        theta = 0
        R_z = np.array([[math.cos(theta), -math.sin(theta), 0],
                        [math.sin(theta), math.cos(theta), 0],
                        [0, 0, 1]
                        ])

        raw_grid = (R_z @ raw_grid.T).T

        raw_grid = raw_grid[exchange, :]

        # shape: (3,4+1)
        self.pts_in_vive_tracker_coord = np.concatenate([raw_grid, np.zeros([1, 3])], axis=0).T


class TrackerDataset:
    def __init__(self, jsonl_path: str):
        self.ts_list = []
        self.pose_list = []
        self.ts_pose = dict()
        with open(jsonl_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                self.ts_list.append(data[0]["sys_ts_ns"])
                self.pose_list.append(data[1])

                self.ts_pose[data[0]["sys_ts_ns"]] = data[1]

            self.start_ts = self.ts_list[0]
            self.end_ts = self.ts_list[-1]

        self.intepolation_funcs = {
            "linear": self.get_interpolation_linear,
        }
        self.hand_board = HandBoard()

    def __getitem__(self, mes: Tuple[float, HandAruco, str]) -> Tuple[Optional[dict[str, np.ndarray]], Optional[str]]:
        """

        :param mes: tuple(ts: float, hand_aruco: HandAruco, itype: str = "linear")
        :return: dict( ["T_vec"] ["Q_vec"] ["R_matrix"] ["pts"] ) or None   ,  str(debug message) or None
        """
        ts = mes[0]
        hand_aruco = mes[1]
        itype = mes[2]

        if ts < self.start_ts or ts > self.end_ts:
            return None, "start_ts: " + self.start_ts + "; end_ts: " + self.end_ts + "; current_ts: " + ts
        else:
            if itype not in self.intepolation_funcs.keys():
                return None, "itype not found"
            return self.intepolation_funcs[itype](ts, hand_aruco)

    def get_interpolation_linear(self, ts: float, hand_aruco: HandAruco) -> Tuple[Optional[dict[str, np.ndarray]], Optional[str]]:
        for i in range(len(self.ts_list) - 1):
            if self.ts_list[i] <= ts < self.ts_list[i + 1]:
                a = ts - self.ts_list[i]
                b = self.ts_list[i + 1] - ts
                if hand_aruco.name in self.pose_list[i].keys() and hand_aruco.name in self.pose_list[i + 1].keys():
                    t1 = np.array(self.pose_list[i][hand_aruco.name][:3])
                    t2 = np.array(self.pose_list[i + 1][hand_aruco.name][:3])
                    q1 = np.array(self.pose_list[i][hand_aruco.name][3:7])
                    q2 = np.array(self.pose_list[i + 1][hand_aruco.name][3:7])

                    t = (t1 * b + t2 * a) / (a + b)
                    print(i)
                    q = (q1 * b + q2 * a) / (a + b)
                    q = q / np.linalg.norm(q)
                    res = dict()
                    res["T_vec"] = t.reshape(3, 1)
                    res["Q_vec"] = q
                    res["R_matrix"] = t3d.quaternions.quat2mat(q)
                    res["pts"] = res["R_matrix"] @ hand_aruco.pts_in_vive_tracker_coord + res["T_vec"]
                    return res, None
                return None, "hand_aruco not found"


def q_cross(q1, q2):
    return np.array([
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[3] * q2[2] - q1[2] * q2[3],
        q1[0] * q2[2] + q1[2] * q2[0] + q1[1] * q2[3] - q1[3] * q2[1],
        q1[0] * q2[3] + q1[3] * q2[0] + q1[2] * q2[1] - q1[1] * q2[2]
    ])


if __name__ == '__main__':
    tracker_data = TrackerDataset(r"C:\Users\robotflow\Desktop\kva-system\tracker_data\test22_20230708_201421.jsonl")

    system_timestamp_offset = json.load(open(r"C:\Users\robotflow\Desktop\kva-system\azure_kinect_data\test22_20230708_201421\kinect/meta.json"))["system_timestamp_offset"]
    ts = 967133 * 1e3 + (system_timestamp_offset + 28800) * 1e9
    print(ts)

    ts = tracker_data.ts_list[347]

    left_hand_aruco = HandAruco("LHR-656F5409", exchange=[1, 2, 3, 0])
    right_hand_aruco = HandAruco("LHR-6A736404")
    pack = tracker_data[(ts, right_hand_aruco, "linear")][0]
    #print(pack["T_vec"])
    _pos = pack["pts"].T
    _pos2 = tracker_data[(ts, left_hand_aruco, "linear")][0]["pts"].T

    coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    transformed_pcd_all = o3d.geometry.PointCloud()
    res_list=[]
    for i, _ in enumerate(tracker_data.ts_list[:320]):
        t1 = np.array(tracker_data.pose_list[i]["LHR-6A736404"][:3])
        r1 = t3d.quaternions.quat2mat(tracker_data.pose_list[i]["LHR-6A736404"][3:])
        pos = (r1 @ right_hand_aruco.pts_in_vive_tracker_coord + t1.reshape(3, 1)).T
        t2 = np.array(tracker_data.pose_list[i]["LHR-656F5409"][:3])
        r2 = t3d.quaternions.quat2mat(tracker_data.pose_list[i]["LHR-656F5409"][3:])
        pos2 = (r2 @ left_hand_aruco.pts_in_vive_tracker_coord + t2.reshape(3, 1)).T
        res_list.append(pos)
        res_list.append(pos2)

    transformed_pcd_all.points = o3d.utility.Vector3dVector(np.concatenate([_ for _ in res_list], axis=0))
    o3d.visualization.draw_geometries([coordinate, transformed_pcd_all])



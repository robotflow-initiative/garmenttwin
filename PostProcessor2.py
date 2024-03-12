import os
import json


class PostProcessor2:
    def __init__(self, res_json_path: str):
        self.res_json_path = res_json_path
        self.camera_info_json_path = os.path.join(os.path.dirname(res_json_path), 'cameras_info.jsonl')

        self.res_json = json.load(open(self.res_json_path, 'r'))
        self.camera_info_json = json.load(open(self.camera_info_json_path, 'r'))
        self.camera_extrinsics = {name: self.camera_info_json[name]['extrinsic'] for name in self.camera_info_json.keys()}

        print(self.res_json)

    def run(self):
        for frame_id in self.res_json.keys():
            self.complete_in_frame(frame_id)
        self.complete_with_other_frame()
        self.save_json()

    def complete_in_frame(self, frame_id: str):
        frame_mes = self.res_json[frame_id].copy()
        for camera_name in self.camera_extrinsics.keys():
            if camera_name not in frame_mes.keys():
                frame_mes[camera_name] = {"left_hand": {"T": [], "Q": [], "R": []}, "right_hand": {"T": [], "Q": [], "R": []}}
            else:
                if "left_hand" not in frame_mes[camera_name].keys():
                    frame_mes[camera_name]["left_hand"] = {"T": [], "Q": [], "R": []}
                if "right_hand" not in frame_mes[camera_name].keys():
                    frame_mes[camera_name]["right_hand"] = {"T": [], "Q": [], "R": []}

        for camera_name in self.camera_extrinsics.keys():
            if len(frame_mes[camera_name]["left_hand"]["T"]) < 3:
                for camera_name2 in self.camera_extrinsics.keys():
                    if len(frame_mes[camera_name2]["left_hand"]["T"]) < 3:
                        continue
                    frame_mes[camera_name]["left_hand"] = frame_mes[camera_name2]["left_hand"].copy()
                    break
            if len(frame_mes[camera_name]["right_hand"]["T"]) < 3:
                for camera_name2 in self.camera_extrinsics.keys():
                    if len(frame_mes[camera_name2]["right_hand"]["T"]) < 3:
                        continue
                    frame_mes[camera_name]["right_hand"] = frame_mes[camera_name2]["right_hand"].copy()
                    break
        self.res_json[frame_id] = frame_mes

    def complete_with_other_frame(self):
        keys = [int(_id) for _id in self.res_json.keys()]
        keys.sort()
        for frame_id in keys:
            if str(frame_id - 3) not in self.res_json.keys(): continue
            for camera_name in self.camera_extrinsics.keys():
                if len(self.res_json[str(frame_id)][camera_name]["left_hand"]["T"]) < 3:
                    if not len(self.res_json[str(frame_id - 3)][camera_name]["left_hand"]["T"]) < 3:
                        self.res_json[str(frame_id)][camera_name]["left_hand"] = self.res_json[str(frame_id - 3)][camera_name]["left_hand"].copy()
                if len(self.res_json[str(frame_id)][camera_name]["right_hand"]["T"]) < 3:
                    if not len(self.res_json[str(frame_id - 3)][camera_name]["right_hand"]["T"]) < 3:
                        self.res_json[str(frame_id)][camera_name]["right_hand"] = self.res_json[str(frame_id - 3)][camera_name]["right_hand"].copy()

        for frame_id in keys[::-1]:
            if str(frame_id + 3) not in self.res_json.keys(): continue
            for camera_name in self.camera_extrinsics.keys():
                if len(self.res_json[str(frame_id)][camera_name]["left_hand"]["T"]) < 3:
                    if not len(self.res_json[str(frame_id + 3)][camera_name]["left_hand"]["T"]) < 3:
                        self.res_json[str(frame_id)][camera_name]["left_hand"] = self.res_json[str(frame_id + 3)][camera_name]["left_hand"].copy()
                if len(self.res_json[str(frame_id)][camera_name]["right_hand"]["T"]) < 3:
                    if not len(self.res_json[str(frame_id + 3)][camera_name]["right_hand"]["T"]) < 3:
                        self.res_json[str(frame_id)][camera_name]["right_hand"] = self.res_json[str(frame_id + 3)][camera_name]["right_hand"].copy()

    def save_json(self):
        with open(os.path.join(os.path.dirname(self.res_json_path), 'res2.json'), 'w') as f:
            json.dump(self.res_json, f, indent=4)


if __name__ == '__main__':
    # pp = PostProcessor2(r'E:\Processed\dress-long-sleeve\dress-long-sleeve_001_20230731_134137\res.json')
    # pp.run()

    for root, dirs, files in os.walk(r"E:"):
        if "res.json" in files:
            pp = PostProcessor2(os.path.join(root, "res.json"))
            pp.run()
            print(os.path.join(root, "res.json"))
        # if root == "E:8-2":
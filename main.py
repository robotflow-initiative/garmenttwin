import glob
import gc
import concurrent.futures
import os
import os.path as osp
import threading

import tqdm

from PostProcessor import PostProcessor
from decoder.StateMachine import StateMachine, state_machine_save_thread_v1
from decoder.wrapper import mkv_record_wrapper
from thirdparty import pykinect
from typing import Dict, Optional, List, Tuple


def mkv_worker(processor: PostProcessor):
    kinect_dir = osp.join(processor.dataset_path, "kinect")
    names = [processor.main_camera_name] + processor.sub_camera_names
    wrappers = [
        mkv_record_wrapper(kinect_dir + "/" + n + ".mkv", processor) for n in names
    ]
    print("wrappers:", [kinect_dir + "/" + n + ".mkv" for n in names])
    m = StateMachine(names)
    t = threading.Thread(target=state_machine_save_thread_v1, args=(m, processor, kinect_dir, names))
    t.start()
    with tqdm.tqdm() as pbar:
        num_closed = 0
        while num_closed < len(wrappers):
            frame_futures: Dict[str, Optional[concurrent.futures.Future]] = {k: None for k in names}
            for idx, w in enumerate(wrappers):
                try:
                    # noinspection PyTypeChecker
                    frame_futures[names[idx]], err = next(w)
                    if err is not None:
                        raise err
                except StopIteration:
                    num_closed += 1
                    continue

            frames = {k: frame_futures[k].result() for k in names if frame_futures[k] is not None}
            # print({k: v['color_dev_ts_usec'] if v is not None else None for k, v in frames.items()})

            for stream_id, frame in frames.items():
                if frame is not None:
                    m.push(stream_id, frame)
                    pbar.set_description(f"pressure: {len(m.frame_buffer[names[-1]])}")
                    pbar.update(1)
        m.close()
        t.join()
    processor.save()
    return


if __name__ == '__main__':
    pykinect.initialize_libraries()
    # PP = PostProcessor(dataset_path=r"E:\ClothPose7-24-after\pants-long_005_20230724_153302",
    #                    target_path=r"E:\ClothPose7-24-PostProcess")
    # mkv_worker(PP)

    for root, dirs, files in os.walk(r"E:"):
        print(root)
        if root == "E:8-2":
            print(root)
            print(dirs)
            Threads = []
            for dir in dirs:
                if dir.split('_')[0] == "cali": continue
                print("processing: ", dir)
                PP = PostProcessor(dataset_path=osp.join(root, dir),
                                   target_path=r"E:\8-2-pp")

                mkv_worker(PP)
            break

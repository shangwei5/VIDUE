import os
import argparse
import cv2
import numpy as np
# import ipdb
from shutil import rmtree, move, copy

parser = argparse.ArgumentParser()
parser.add_argument('--videos_src_path', type=str, default='/mnt/disk10T/shangwei/data/adobe240/test', help='the dir of high-frame-rate video')
parser.add_argument('--videos_save_path', type=str, default='/mnt/disk10T/shangwei/data/adobe240/LFR_Adobe_53', help='the dir of low-frame-rate video')
parser.add_argument('--num_compose', type=int, default=5*3, help='how many frames to compose a blurry frame')
parser.add_argument('--tot_inter_frame', type=int, default=8*3, help='the down sample rate')

args = parser.parse_args()


def generate_blur():
    videos_save_path = args.videos_save_path
    videos_src_path = args.videos_src_path
    num_compose = args.num_compose
    videos = os.listdir(videos_src_path)
    videos = sorted(videos)
    if not os.path.exists(videos_save_path):
        os.mkdir(videos_save_path, True)
    cnt = 0
    for video in videos:
        print("generate low-frame-rate video from video:%s" % video)
        cnt += 1
        
        save_path = os.path.join(videos_save_path, video)
        src_path = os.path.join(videos_src_path, video)

        frames = os.listdir(src_path)
        frames = sorted(frames)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        num_frames = len(frames)
        tot_inter_frame = args.tot_inter_frame
        num_batch = num_frames // tot_inter_frame

        for i in range(num_batch):
            frames_to_compose = frames[i * tot_inter_frame:i * tot_inter_frame + num_compose]
            out_frame_name = frames[i * tot_inter_frame + num_compose // 2]  #"{:06d}.png".format(i * tot_inter_frame + num_compose // 2)
            
            gt_img = cv2.imread(os.path.join(src_path, frames_to_compose[0]))
            H, W, C = gt_img.shape

            blur_img = np.zeros((H//2, W//2, C), dtype=np.float32)
            for j in range(num_compose):
                frame_name = os.path.join(src_path, frames_to_compose[j])
                frame_j = cv2.imread(frame_name)
                frame_j = cv2.resize(frame_j, None, fx=0.5, fy=0.5)
                blur_img += frame_j
                
            blur_img = blur_img / num_compose
            print("generate blur frame:", out_frame_name)
            cv2.imwrite(os.path.join(save_path, out_frame_name), blur_img)



if __name__ == "__main__":
    generate_blur()

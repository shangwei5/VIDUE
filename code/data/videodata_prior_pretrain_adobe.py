import os
import glob
import numpy as np
import imageio
import torch
import torch.utils.data as data
import utils.utils as utils
import cv2
import random

class VIDEODATA(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.in_seq = args.n_sequence
        self.n_seq = args.n_sequence * args.n_outputs
        self.n_frames_per_video = args.n_frames_per_video
        self.xN = args.n_outputs   #
        self.random = args.random
        self.batch = args.batch_size
        if self.random:
            self.exposure = [(i+1) * self.args.blur_deg for i in range(self.xN)]
            self.readout = [self.xN * self.args.blur_deg-j for j in self.exposure]
            if self.xN > 4:
                self.exposure = [(i+1) * self.args.blur_deg for i in range(0,self.xN,2)]
                self.readout = [self.xN * self.args.blur_deg - j for j in self.exposure]
            self.curr_exposure = self.exposure[0]
        else:
            self.exposure = args.m * self.args.blur_deg
            self.readout = args.n * self.args.blur_deg
        print("n_seq:", self.n_seq)
        print("exposure:", self.exposure, "readout:", self.readout)
        print("n_frames_per_video:", args.n_frames_per_video)

        self.n_frames_video = []
        if train:
            self.apath = args.dir_data
        else:
            self.apath = args.dir_data_test

        self.images = self._scan()
        # random.shuffle(self.images)

        self.num_video = len(self.images)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq * self.args.blur_deg - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

        if train:
            self.repeat = max(args.test_every // max((self.num_frame // self.args.batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

        if args.process:
            self.data_images = self._load(self.images)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data

    def _scan(self):
        if self.train:
            vid_names = sorted(glob.glob(os.path.join(self.apath, '*')))    #[0:1]
        else:
            vid_names = sorted(glob.glob(os.path.join(self.apath, '*')))[:1]

        images = []

        for i in range(len(vid_names)):
            if self.train:
                dir_names = sorted(glob.glob(os.path.join(vid_names[i], '*')))[:self.args.n_frames_per_video]
            else:
                dir_names = sorted(glob.glob(os.path.join(vid_names[i], '*')))
            images.append(dir_names)
            self.n_frames_video.append(len(dir_names))

        return images

    def _load(self, images):
        data_images = []

        n_videos = len(images)
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" % idx)
            gts = [imageio.imread(hr_name) for hr_name in images[idx]]
            gts = np.array([cv2.resize(frame, None, fx=0.5, fy=0.5) for frame in gts])
            print(gts.shape)
            data_images.append(gts)

        return data_images

    def __getitem__(self, idx):
        # inputs, label, input_filenames, self.curr_exposure
        if self.args.process:
            inputs, label, input_filenames, exposure = self._load_file_from_loaded_data(idx)
        else:
            inputs, label, input_filenames, exposure = self._load_file(idx)

        inputs_list = [inputs[i, :, :, :] for i in range(self.n_seq // self.xN)]
        inputs_concat = np.concatenate(inputs_list, axis=2)
        # gts_list = [gts[i, :, :, :] for i in range(self.xN)]
        # gts_concat = np.concatenate(gts_list, axis=2)

        inputs_concat = self.get_patch(inputs_concat, self.args.size_must_mode)  #, bms_concat     , bms_concat
        # inputs_list = [inputs_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.n_seq // self.xN)]
        # gts_list = [gts_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.xN)]
        # print(inputs_concat[0].shape, inputs_concat[1].shape)
        inputs = np.array(inputs_concat)
        # print(inputs.shape)
        # gts = np.array(gts_list)
        # h,w,_ = gts[0].shape
        # prior = [1] * exposure + [0] * (self.xN - exposure)
        # prior = np.array(prior)
        # prior = prior.reshape(-1, 1, 1)
        # prior = np.tile(prior, (h, w))

        input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        # gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        # print(label)
        # print(label.shape)
        label_tensors = torch.from_numpy(label).float()

        return input_tensors, label_tensors, input_filenames, exposure

    def __len__(self):
        if self.train:
            return self.num_frame #* self.repeat
        else:
            return self.num_frame // (self.xN * self.args.blur_deg) + 1

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_frame
        else:
            return idx * self.xN

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        if self.random:
            # if idx % self.batch == 0:
            #     self.curr_exposure = random.choice(self.exposure)
            # idx = np.random.randint(0, self.num_frame)
            label = np.random.randint(0, len(self.exposure))
            self.curr_exposure = self.exposure[label]
            label = np.array(label)
            # print(idx, self.curr_exposure)
            # for ij in range(len(self.images)):
            #     last = self.images[ij][-self.n_seq:]
            #     random.shuffle(self.images[ij])
        else:
            self.curr_exposure = self.exposure

        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq*self.args.blur_deg + 1 for n in self.n_frames_video]
        # random.shuffle(n_poss_frames)
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        # if self.in_seq % 2 == 0:
        #     f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq // 2-1):frame_idx + self.xN * (self.in_seq // 2)]
        # else:
        #     f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq//2):frame_idx + self.xN * (self.in_seq//2 +1)]
        # f_inputs = self.images[video_idx][frame_idx+self.curr_exposure//2:frame_idx + self.n_seq:self.xN]
        f_inputs = self.images[video_idx][frame_idx:frame_idx + self.n_seq*self.args.blur_deg:self.xN*self.args.blur_deg]
        inputs = []
        frame = imageio.imread(f_inputs[0])
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        img = np.array(frame)
        H, W, C = img.shape
        for i in range(frame_idx,frame_idx + self.n_seq*self.args.blur_deg,self.xN*self.args.blur_deg):
            blur = np.zeros((H, W, C))
            for j in range(i,i+self.curr_exposure):
                frame_j = imageio.imread(self.images[video_idx][j])
                frame_j = cv2.resize(frame_j, None, fx=0.5, fy=0.5)
                blur += np.array(frame_j) / self.curr_exposure
            inputs.append(blur)

        # gts = [imageio.imread(hr_name) for hr_name in f_gts]
        # gts = np.array([cv2.resize(gt, None, fx=0.5, fy=0.5) for gt in gts])
        inputs = np.array(inputs)
        # output_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
        #              for name in f_gts]
        input_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                            for name in f_inputs]

        return inputs, label, input_filenames, self.curr_exposure  #gts, output_filenames,

    def _load_file_from_loaded_data(self, idx):
        if self.random:
            # kinds = len(self.exposure)
            # exposure = self.exposure[idx%kinds]
            label = np.random.randint(0, len(self.exposure))
            self.curr_exposure = self.exposure[label]
            label = np.array(label)
        else:
            exposure = self.exposure

        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq*self.args.blur_deg + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        # if self.in_seq % 2 == 0:
        #     f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq // 2-1):frame_idx + self.xN * (self.in_seq // 2)]
        # else:
        #     f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq//2):frame_idx + self.xN * (self.in_seq//2 +1)]
        # f_inputs = self.images[video_idx][frame_idx+exposure//2:frame_idx + self.n_seq:self.xN]
        f_inputs = self.images[video_idx][frame_idx:frame_idx + self.n_seq*self.args.blur_deg:self.xN*self.args.blur_deg]
        H, W, C = self.data_images[0][0].shape
        inputs = []
        for i in range(frame_idx, frame_idx + self.n_seq*self.args.blur_deg, self.xN*self.args.blur_deg):
            blur = np.zeros((H, W, C))
            for j in range(i, i + self.curr_exposure):
                frame_j = self.data_images[video_idx][j]
                blur += np.array(frame_j) / self.curr_exposure
            inputs.append(blur)

        # gts = [imageio.imread(hr_name) for hr_name in f_gts]
        # gts = np.array([cv2.resize(gt, None, fx=0.5, fy=0.5) for gt in gts])
        inputs = np.array(inputs)
        # output_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
        #              for name in f_gts]
        input_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                            for name in f_inputs]

        return inputs, label, input_filenames, self.curr_exposure


    def get_patch(self, input, size_must_mode=1):   #, bm
        if self.train:
            input1 = utils.get_patch(input, patch_size=self.args.patch_size)   #, bm    , bm
            input2 = utils.get_patch(input, patch_size=self.args.patch_size)
            h, w, c = input1.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input1, input2 = input1[:new_h, :new_w, :], input2[:new_h, :new_w, :]   #, bm     , bm[:new_h, :new_w, :]
            if not self.args.no_augment:
                input = utils.data_augment(input1,input2)
        else:
            input1 = utils.get_patch(input, patch_size=self.args.patch_size)  # , bm    , bm
            input2 = utils.get_patch(input, patch_size=self.args.patch_size)
            h, w, c = input1.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input1, input2 = input1[:new_h, :new_w, :], input2[:new_h, :new_w, :]  # , bm     , bm[:new_h, :new_w, :]
            if not self.args.no_augment:
                input = utils.data_augment(input1,input2)
        return input

"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

import numpy as np
#import ffmpeg
#import skvideo.io
import pandas as pd
#from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import accimage
import pdb


def image_to_np(image):
  image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
  image.copyto(image_np)
  image_np = np.transpose(image_np, (1,2,0))
  return image_np


def readim(image_name):
  # read image
  img_data = accimage.Image(image_name)
  img_data = image_to_np(img_data) # RGB
  return img_data


def load_from_frames(foldername, framenames, start_index, tuple_len, clip_len, interval):
  clip_tuple = []
  for i in range(tuple_len):
      one_clip = []
      for j in range(clip_len):
          im_name = os.path.join(foldername, framenames[start_index + i * (tuple_len + interval) + j])
          im_data = readim(im_name)
          one_clip.append(im_data)
      #one_clip_arr = np.array(one_clip)
      clip_tuple.append(one_clip)
  return clip_tuple


def load_one_clip(foldername, framenames, start_index, clip_len):
    one_clip = []
    for i in range(clip_len):
        im_name = os.path.join(foldername, framenames[start_index + i])
        im_data = readim(im_name)
        one_clip.append(im_data)

    return np.array(one_clip)


class HMDB51Dataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir, clip_len=16, split='1', train=True, val=False, transforms_=None, test_sample_num=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'hmdb51', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'hmdb51', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'hmdb51', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split'+ self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]

        class_idx = self.class_label2idx[videoname[:videoname.find('/')]] - 1 # add - 1 because it is range [1,101] which should be [0, 100]
        '''
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        '''
        # videoname = 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]
        # to avoid void folder
        vids = vid.split('_')
        if vids[1] == 'HandStandPushups':
            vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]

        #framefolder = os.path.join('/raid/home/taoli/dataset/hmdb51/jpegs_256', vid)
        framefolder = os.path.join('/raid/dataset/hmdb51/jpegs_256', vid) # in aoba
        #framefolder = os.path.join('/work/taoli/hmdb51_rgbflow/jpegs_256/', vid) # + v_**

        filenames = ['frame000001.jpg']
        for parent, dirnames, filenames in os.walk(framefolder):
            if 'n_frames' in filenames:
                filenames.remove('n_frames')
            filenames = sorted(filenames)
        framenames = filenames
        length = len(framenames)
        '''
        if length < 16:
            print(vid, length)
            print('\n')
            raise
        '''
        # random select a clip for train
        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = load_one_clip(framefolder, framenames, clip_start, self.clip_len)
            #clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))
        # sample several clips for test
        else:
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                clip_start = int(i - self.clip_len/2)
                #clip = videodata[clip_start: clip_start + self.clip_len]
                clip = load_one_clip(framefolder, framenames, clip_start, self.clip_len)
                if self.transforms_:
                    trans_clip = []
                    # fix seed, apply the sample `random transformation` for all frames in the clip 
                    seed = random.random()
                    for frame in clip:
                        random.seed(seed)
                        frame = self.toPIL(frame) # PIL image
                        frame = self.transforms_(frame) # tensor [C x H x W]
                        trans_clip.append(frame)
                    # (T x C X H x W) to (C X T x H x W)
                    clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                else:
                    clip = torch.tensor(clip)
                all_clips.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))

            return torch.stack(all_clips), torch.tensor(int(class_idx))

class HMDB51ClipRetrievalDataset(Dataset):
    """UCF101 dataset for Retrieval. Sample clips for each video. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        sample_num(int): number of clips per video.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, sample_num, train=True, transforms_=None, split='1'):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.sample_num = sample_num
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'hmdb51', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        self.split = split

        if self.train:
            train_split_path = os.path.join(root_dir, 'hmdb51', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'hmdb51', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        '''
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        '''
        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]
        # to avoid void folder
        vids = vid.split('_')
        if vids[1] == 'HandStandPushups':
            vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]

        #framefolder = os.path.join('/raid/home/taoli/dataset/hmdb51/jpegs_256/', vid) # + v_**
        framefolder = os.path.join('/work/taoli/hmdb51_rgbflow/jpegs_256/', vid) # + v_**

        filenames = ['frame000001.jpg']
        for parent, dirnames, filenames in os.walk(framefolder):
            if 'n_frames' in filenames:
                filenames.remove('n_frames')
            filenames = sorted(filenames)
        framenames = filenames
        length = len(framenames)

        all_clips = []
        all_idx = []
        for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.sample_num):
            clip_start = int(i - self.clip_len/2)
            #clip = videodata[clip_start: clip_start + self.clip_len]
            clip = load_one_clip(framefolder, framenames, clip_start, self.clip_len)
            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)
            all_clips.append(clip)
            all_idx.append(torch.tensor(int(class_idx)))

        return torch.stack(all_clips), torch.stack(all_idx)


class UCF101VCOPDataset(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len #clip_len * tuple_len + interval * (tuple_len - 1)

        if self.train:
            vcop_train_split_name = 'vcop_train_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_train_split_path = os.path.join(root_dir, 'split', vcop_train_split_name)
            self.train_split = pd.read_csv(vcop_train_split_path, header=None)[0]
            self.train_len = len(self.train_split)
        else:
            vcop_test_split_name = 'vcop_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_test_split_path = os.path.join(root_dir, 'split', vcop_test_split_name)
            self.test_split = pd.read_csv(vcop_test_split_path, header=None)[0]
            self.test_len = len(self.test_split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname1 = self.train_split[idx]
            other_idx = (idx + random.randint(0, self.train_len - 1)) % self.train_len
            videoname2 = self.train_split[other_idx]
        else:
            videoname1 = self.test_split[idx]
            other_idx = (idx + random.randint(0, self.test_len - 1)) % self.train_len
            videoname2 = self.test_split[other_idx]
        
        def get_frame_list(videoname):
            framefolder = os.path.join('/home3/taoli/workspace/ssl/data/frames', videoname[:-4])
            for parent, dirnames, filenames in os.walk(framefolder):
                if 'n_frames' in filenames:
                    filenames.remove('n_frames')
                filenames = sorted(filenames)
                return framefolder, filenames

        # pos and anchor
        framefolder1, framenames1 = get_frame_list(videoname1)
        length1 = len(framenames1)
        # neg
        framefolder2, framenames2 = get_frame_list(videoname2)
        length2 = len(framenames2)

        if self.train:
            anchor_start = random.randint(0, length1 - self.tuple_total_frames)
            pos_start = random.randint(0, length1 - self.tuple_total_frames)
            neg_start = random.randint(0, length2 - self.tuple_total_frames)
        else:
            random.seed(idx)
            anchor_start = random.randint(0, length1 - self.tuple_total_frames)
            pos_start = random.randint(0, length1 - self.tuple_total_frames)
            neg_start = random.randint(0, length2 - self.tuple_total_frames)
        
        pos_clip = load_one_clip(framefolder1, framenames1, pos_start, self.clip_len)
        anchor_clip = load_one_clip(framefolder1, framenames1, pos_start, self.clip_len)
        neg_clip = load_one_clip(framefolder1, framenames1, pos_start, self.clip_len)

        tuple_clip = [anchor_clip, pos_clip, neg_clip]

        if self.transforms_:
            trans_tuple = []
            for clip in tuple_clip:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image, this is activated when using skvideo.io
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return torch.stack(tuple_clip)


class UCF101FrameRetrievalDataset(Dataset):
    """UCF101 dataset for Retrieval. Sample frames for each video. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        sample_num(int): number of clips per video.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, sample_num, train=True, transforms_=None):
        self.root_dir = root_dir
        self.sample_num = sample_num
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist01.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist01.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
        all_frames = []
        all_idx = []
        for i in np.linspace(0, length-1, self.sample_num):
            frame = videodata[int(i)]
            if self.transforms_:
                frame = self.toPIL(frame) # PIL image
                frame = self.transforms_(frame) # tensor [C x H x W]
            else:
                frame = torch.tensor(frame) 
            all_frames.append(frame)
            all_idx.append(torch.tensor(int(class_idx)))

        return torch.stack(all_frames), torch.stack(all_idx)



def export_tuple(tuple_clip, tuple_order, dir):
    """export tuple_clip and set its name with correct order.
    
    Args:
        tuple_clip (tensor): [tuple_len x channel x time x height x width]
        tuple_order (tensor): [tuple_len]
    """
    tuple_len, channel, time, height, width = tuple_clip.shape
    for i in range(tuple_len):
        filename = os.path.join(dir, 'c{}.mp4'.format(tuple_order[i]))
        skvideo.io.vwrite(filename, tuple_clip[i])


def gen_ucf101_vcop_splits(root_dir, clip_len, interval, tuple_len):
    """Generate split files for different configs."""
    vcop_train_split_name = 'vcop_train_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    vcop_train_split_path = os.path.join(root_dir, 'split', vcop_train_split_name)
    vcop_test_split_name = 'vcop_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    vcop_test_split_path = os.path.join(root_dir, 'split', vcop_test_split_name)
    # minimum length of video to extract one tuple
    min_video_len = clip_len * tuple_len + interval * (tuple_len - 1)

    def _video_longer_enough(filename):
        """Return true if video `filename` is longer than `min_video_len`"""
        path = os.path.join(root_dir, 'video', filename)
        metadata = ffprobe(path)['video']
        return eval(metadata['@nb_frames']) >= min_video_len

    train_split = pd.read_csv(os.path.join(root_dir, 'split', 'trainlist01.txt'), header=None, sep=' ')[0]
    train_split = train_split[train_split.apply(_video_longer_enough)]
    train_split.to_csv(vcop_train_split_path, index=None)

    test_split = pd.read_csv(os.path.join(root_dir, 'split', 'testlist01.txt'), header=None, sep=' ')[0]
    test_split = test_split[test_split.apply(_video_longer_enough)]
    test_split.to_csv(vcop_test_split_path, index=None)


def ucf101_stats():
    """UCF101 statistics"""
    collects = {'nb_frames': [], 'heights': [], 'widths': [], 
                'aspect_ratios': [], 'frame_rates': []}

    for filename in glob('../data/ucf101/video/*/*.avi'):
        metadata = ffprobe(filename)['video']
        collects['nb_frames'].append(eval(metadata['@nb_frames']))
        collects['heights'].append(eval(metadata['@height']))
        collects['widths'].append(eval(metadata['@width']))
        collects['aspect_ratios'].append(metadata['@display_aspect_ratio'])
        collects['frame_rates'].append(eval(metadata['@avg_frame_rate']))

    stats = {key: sorted(list(set(collects[key]))) for key in collects.keys()}
    stats['nb_frames'] = [stats['nb_frames'][0], stats['nb_frames'][-1]]

    pprint(stats)


if __name__ == '__main__':
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ucf101_stats()
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 16, 2)
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 32, 3)
    gen_ucf101_vcop_splits('../data', 16, 8, 3)

    # train_transforms = transforms.Compose([
    #     transforms.Resize((128, 171)),
    #     transforms.RandomCrop(112),
    #     transforms.ToTensor()])
    # # train_dataset = UCF101FOPDataset('../data/ucf101', 8, 3, True, train_transforms)
    # # train_dataset = UCF101VCOPDataset('../data/ucf101', 16, 8, 3, True, train_transforms)
    # train_dataset = UCF101Dataset('../data/ucf101', 16, False, train_transforms)
    # # train_dataset = UCF101RetrievalDataset('../data/ucf101', 16, 10, True, train_transforms)    
    # train_dataloader = DataLoader(train_dataset, batch_size=8)

    # for i, data in enumerate(train_dataloader):
    #     clips, idxs = data
    #     # for i in range(10):
    #     #     filename = os.path.join('{}.mp4'.format(i))
    #     #     skvideo.io.vwrite(filename, clips[0][i])
    #     print(clips.shape)
    #     print(idxs)
    #     exit()
    # pass

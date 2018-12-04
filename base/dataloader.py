import sys
sys.path.append('..')
import json
import pickle as pkl
import numpy as np
import random
import os
import nltk
import h5py
from gensim.models import KeyedVectors
import torch
from torch.utils.data import Dataset, DataLoader
from criteria import calculate_IoU, calculate_nIoL


def load_file(filename):
    with open(filename,'rb') as fr:
        return pkl.load(fr)


def load_json(filename):
    with open(filename) as fr:
        return json.load(fr)


class Loader(Dataset):
    def __init__(self, params, key_file, word2vec, flag=False):
        #general
        self.params = params
        self.is_training = flag
        self.feature_path = params['feature_path']
        self.feature_path_tsn = params['feature_path_tsn']

        self.max_batch_size = params['batch_size']

        # dataset
        self.word2vec = word2vec
        self.key_file = load_json(key_file)
        self.dataset_size = len(self.key_file)


        # frame / question
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']


    def __getitem__(self, index):

        frame_vecs = np.zeros((self.max_frames, self.input_video_dim), dtype=np.float32)

        ques_vecs = np.zeros((self.max_words, self.input_ques_dim), dtype=np.float32)

        keys = self.key_file[index]
        vid, duration, timestamps, sent = keys[0], keys[1], keys[2], keys[3]

        # video
        if self.params['is_origin_dataset']:
            if not os.path.exists(self.feature_path + '/%s.h5' % vid):
                print('the video is not exist:', vid)
            with h5py.File(self.feature_path + '/%s.h5' % vid, 'r') as fr:
                feats = np.asarray(fr['feature'])
        else:
            vid = vid[2:]
            while not os.path.exists(self.feature_path_tsn + '/feat/%s.h5' % vid):
                keys = self.key_file[0]
                vid, duration, timestamps, sent = keys[0], keys[1], keys[2], keys[3]
                vid = vid[2:]

            with h5py.File(self.feature_path_tsn + '/feat/%s.h5' % vid, 'r') as hf:
                fg = np.asarray(hf['fg'])
                bg = np.asarray(hf['bg'])
                feat = np.hstack([fg, bg])
            with h5py.File(self.feature_path_tsn + '/flow/%s.h5' % vid, 'r') as hf:
                fg2 = np.asarray(hf['fg'])
                bg2 = np.asarray(hf['bg'])
                feat2 = np.hstack([fg2, bg2])
            feats = feat + feat2

        inds = np.int32(np.linspace(start=0, stop=feats.shape[0] - 1, num=self.max_frames))
        frames = feats[inds, :]
        frames = np.vstack(frames)
        frame_vecs[:self.max_frames, :] = frames[:self.max_frames, :]
        frame_n = np.array(len(frame_vecs),dtype=np.int32)

        # [32,64,128,256] / [8,16,32,64]
        frame_per_sec = self.max_frames/duration
        start_frame = round(frame_per_sec * timestamps[0])
        end_frame = round(frame_per_sec * timestamps[1]) - 1
        if end_frame < start_frame:
            end_frame = start_frame

        frame_mask = np.ones([self.max_frames], np.int32)
        ques_mask = np.zeros([self.max_words], np.int32)

        start_label = np.zeros([self.max_frames], np.int32)
        start_label[start_frame] = 1
        end_label = np.zeros([self.max_frames], np.int32)
        end_label[end_frame] = 1
        # question
        stopwords = ['.', '?', ',', '']
        sent = nltk.word_tokenize(sent)
        ques = [word.lower() for word in sent if word not in stopwords]
        ques = [self.word2vec[word] for word in ques if word in self.word2vec]
        ques_feats = np.stack(ques, axis=0)
        ques_n = min(len(ques), self.max_words)
        ques_vecs[:ques_n, :] = ques_feats[:ques_n, :]
        ques_mask[range(ques_n)] = 1

        return frame_vecs, frame_mask, ques_vecs, ques_mask, start_frame, end_frame

    def __len__(self):
        if self.is_training:
            return self.dataset_size
        else:
            return 1280


if __name__ == '__main__':
    config = {
        "learning_rate": 1e-3,
        "lr_decay_n_iters": 3000,
        "lr_decay_rate": 0.8,
        "max_epoches": 10,
        "early_stopping": 5,
        "cache_dir": "../results/baseline/",
        "display_batch_interval": 100,
        "evaluate_interval": 5,

        "regularization_beta": 1e-7,
        "dropout_prob": 0.9,

        "batch_size": 64,

        "input_video_dim": 500,
        "max_frames": 384,
        "input_ques_dim": 300,
        "max_words": 20,
        "hidden_size": 512,

        "word2vec": "data/word2vec.bin",

        "is_origin_dataset": True,
        "train_json": "data/activity-net/train.json",
        "val_json": "data/activity-net/val_1.json",
        "test_json": "data/activity-net/val_2.json",
        "train_data": "data/activity-net/train_data.json",
        "val_data": "data/activity-net/val_data.json",
        "test_data": "data/activity-net/test_data.json",
        "feature_path": "data/activity-c3d",
        "feature_path_tsn": "data/tsn_score"
    }

    word2vec = KeyedVectors.load_word2vec_format(config["word2vec"], binary=True)

    train_dataset = Loader(config, config['train_data'], word2vec)


    # Data loader (this provides queues and threads in a very simple way).
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # When iteration starts, queue and thread start to load data from files.
    data_iter = iter(train_loader)

    # Mini-batch images and labels.

    frame_vecs, frame_mask, ques_vecs, ques_mask, starts, ends = data_iter.next()
    print(frame_vecs.dtype)  # float32
    print(frame_mask.dtype)  # int32
    print(ques_vecs.dtype)  # float32
    print(ques_mask.dtype)  # int32
    print(starts)
    print(ends)


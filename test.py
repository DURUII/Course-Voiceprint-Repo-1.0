#! /usr/bin/env python3
import os
import sys
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torchaudio
import model.resnet as model_2d
import model.tdnn as model_1d
import model.classifier as classifiers
from torch.utils.data import DataLoader
from dataset import WavDataset
import torch.nn.functional as F
from config.config_scoring import Config
from torch.utils.data import DataLoader
from scipy import spatial
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Network Parser')
parser.add_argument('--epoch', default=-1, type=int)
args = parser.parse_args()


def main():
    opt = Config()

    if opt.onlyscoring:
        embd_dict = np.load('exp/%s/%s_%s.npy' % (opt.save_dir,
                            opt.save_name, args.epoch), allow_pickle=True).item()
        results = get_results(
            embd_dict, trial_file='data/%s/trials' % opt.val_dir)
        with open("exp/%s/scores_%d.txt" % (opt.save_dir, args.epoch), 'w') as out:
            out.write(results)

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

        # validation dataset
        val_dataset = WavDataset(opt=opt, train_mode=False)
        val_dataloader = DataLoader(val_dataset,
                                    num_workers=opt.workers,
                                    batch_size=1,
                                    pin_memory=True)

        if opt.conv_type == '1D':
            model = getattr(model_1d, opt.model)(in_dim=opt.in_planes, embedding_size=opt.embd_dim,
                                                 hidden_dim=opt.hidden_dim).cuda()  # tdnn, ecapa_tdnn
        elif opt.conv_type == '2D':
            model = getattr(model_2d, opt.model)(
                in_planes=opt.in_planes, embedding_size=opt.embd_dim).cuda()  # resnet

        print('Load exp/%s/model_%d.pkl' % (opt.save_dir, args.epoch))
        checkpoint = torch.load('exp/%s/model_%d.pkl' %
                                (opt.save_dir,  args.epoch))
        model.load_state_dict(checkpoint['model'])
        model = nn.DataParallel(model)

        model.eval()
        embd_dict = {}
        with torch.no_grad():
            for (feat, utt) in tqdm(val_dataloader):
                outputs = model(feat.cuda())
                for i in range(len(utt)):
                    embd_dict[utt[i]] = outputs[i, :].cpu().numpy()
        np.save('exp/%s/%s_%s.npy' %
                (opt.save_dir, opt.save_name, args.epoch), embd_dict)
        print("Embeddings Loaded")
        results = get_results(
            embd_dict, trial_file='data/%s/trials' % opt.val_dir)
        with open("exp/%s/scores_%d.txt" % (opt.save_dir, args.epoch), 'w') as out:
            out.write(results)


def get_results(embd_dict, trial_file, enrol_multi=False, test_multi=True, embd_dim=256, ch=8):
    results = ""
    with open(trial_file) as fh:
        for line in fh:
            line = line.strip()
            utt1, utt2 = line.split()

            if enrol_multi:
                embd1 = np.zeros(embd_dim)
                for i in range(ch):
                    utt1_tmp = utt1+'_{}'.format(str(i))
                    # utt1_tmp = utt1.replace('{}', str(i).zfill(2))
                    embd1 += embd_dict[utt1_tmp]
            else:
                embd1 = embd_dict[utt1]

            if test_multi:
                embd2 = np.zeros(embd_dim)
                for i in range(ch):
                    utt2_tmp = utt2+'_{}'.format(str(i))
                    # utt2_tmp = utt2.replace('{}', str(i).zfill(2))
                    embd2 += embd_dict[utt2_tmp]
            else:
                embd2 = embd_dict[utt2]

            result = 1 - spatial.distance.cosine(embd1, embd2)
            results = results+str(result)+"\n"
    return results


if __name__ == '__main__':
    main()
    print("scores.txt Generated")

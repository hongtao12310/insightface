# coding: utf-8

import face_embedding
import argparse
import cv2
import numpy as np
import os, sys, time, datetime

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r50-am-lfw/model,0', help='path to load model.')
parser.add_argument('--ctx', default='cpu,0', type=str, help='context cpu or gpu')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

age_cache = {"test": [], "train": []}


def walk_dir(dirname, topdown=False):
    for root, dirs, files in os.walk(dirname, topdown=False):
        for f in files:
            yield f, os.path.join(root, f)


def save_numpy_array(path, array):
    np.save(path, array, allow_pickle=False)


def load_age_label(tp, path):
    global age_cache
    with open(path, 'r') as f:
        age_cache[tp] = f.readlines()


def get_age(tp, fid):
    return age_cache[tp][fid-1].strip('\n')


AGE_LABEL = {
    "test": "../datasets/megaage_asian/list/test_age.txt",
    "train": "../datasets/megaage_asian/list/train_age.txt",
}

FACE_DATA = {
    "test": "../datasets/megaage_asian/test",
    "train": "../datasets/megaage_asian/train",
}

VEC_OUTPUT = "../outputs/vectors_50"
IMG_OUTPUT = "../outputs/images"

gender = "X"

if __name__ == '__main__':
    # load age labels into cache
    for tp, path in AGE_LABEL.items():
        load_age_label(tp, path)

    total = 0
    test_index = 0
    train_index = 0
    not_detect = 0
    model = face_embedding.FaceModel(args)

    tbegin = time.time()
    for tp, folder in FACE_DATA.items():
        print("===========dataset %s===========" % tp)
        # find age for a Face
        for f, p in walk_dir(folder):
            total += 1

            if tp == "test":
                test_index += 1
            else:
                train_index += 1
            fid = int(f.strip(".jpg"))
            age = get_age(tp, fid)

            img = cv2.imread(p)
            fName = '_'.join([str(fid), age, gender])
            fPath = os.path.join(VEC_OUTPUT, tp, fName)
            vec = model.get_feature(img, tp, fName)
            if vec is None:
                not_detect += 1
            else:
                save_numpy_array(fPath, vec)

            # if tp == "test" and test_index == 10 \
            #         or tp == "train" and train_index == 10 :
            #     break

    tend = time.time()
    tstr = str(datetime.timedelta(seconds=tend-tbegin))
    print("time eclipse: %s" % tstr)
    print("total faces: %d. only %d faces was not detected" % (total, not_detect))

    # f2 = np.load("./vectors.npy")
        # print(f2.shape)
        # print(type(f2))
        # print(f2[0])

        # np.savetxt("./vectors.out", f1, fmt="%f",  delimiter=',')

        # img = cv2.imread('/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54733.png')
        # f2 = model.get_feature(img)
        # dist = np.sum(np.square(f1-f2))
        # print(dist)
        # sim = np.dot(f1, f2.T)
        # print(sim)
        #diff = np.subtract(source_feature, target_feature)
        #dist = np.sum(np.square(diff),1)

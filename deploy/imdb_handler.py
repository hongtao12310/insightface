# coding: utf-8

import face_embedding
import argparse
import cv2
import numpy as np
import os, sys, time
import datetime
import scipy.io as sio

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r18-amf/model,0', help='path to load model.')
parser.add_argument('--ctx', default='cpu,0', type=str, help='context cpu or gpu')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()


def calc_age(taken, dob):
    birth = datetime.datetime.fromordinal(max(int(dob) - 366, 1))
    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path):
    meta = sio.loadmat(mat_path)
    imdb = meta["imdb"][0, 0]
    dob = imdb[0][0]
    photo_taken = imdb[1][0]
    full_path = imdb[2][0]
    gender = imdb[3][0]
    name = imdb[4][0]
    # face_location = imdb[5][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    return dob, photo_taken, full_path, gender, age


def save_numpy_array(path, array):
    np.save(path, array, allow_pickle=False)


mat_path = "/Users/hongtaozhang/workspace/machine-learning/face_detect/imdb/imdb.mat"
data_path = "/Users/hongtaozhang/workspace/machine-learning/face_detect/imdb/imdb_crop"

vector_output = "../outputs/imdb/vectors"
img_output = "../outputs/imdb/images"


def create_dirs(*args):
    for t in args:
        if not os.path.exists(t):
            os.makedirs(t)


if __name__ == '__main__':
    not_detect = 0
    total = 0
    test_count = 0
    tbegin = time.time()

    model = face_embedding.FaceModel(args)

    create_dirs(vector_output, img_output)
    (dob, photo_token, full_path, gender, age) = get_meta(mat_path)
    for index, x in enumerate(dob):
        total += 1
        test_count += 1
        print("dob: %d" % x)
        print("photo_token: %d" % photo_token[index])
        print("full_path: %s" % full_path[index][0].encode())
        print("gender: %d" % gender[index])
        print("age: %d" % age[index])
        if gender[index] == 0:
            gen = 'F'
        elif gender[index] == 1:
            gen = 'M'
        else:
            gen = 'X'
        # 人脸文件名
        # 文件相对路径 01/nm0000001_rm124825600_1899-5-10_1968.jpg
        fpath = full_path[index][0].encode()

        # image 输入路径
        inpath = os.path.join(data_path, fpath)
        img = cv2.imread(inpath)

        # image 文件名: 原始文件名_age_gender
        fname = fpath.split('/')[-1].strip(".jpg")
        fname = '_'.join([fname, str(age[index]), gen])

        # 输出vector path
        ovec_path = os.path.join(vector_output, fname)
        # 输出image path
        oimage_path = os.path.join(img_output, fname + ".jpg")
        vec = model.get_feature(img, oimage_path)
        print(vec)
        print(type(vec))
        print(vec.shape)
        if vec is None:
            not_detect += 1
        else:
            save_numpy_array(ovec_path, vec)
        if test_count == 10:
            break

    tend = time.time()
    tstr = str(datetime.timedelta(seconds=tend - tbegin))
    print("time eclipse: %s" % tstr)
    print("total faces: %d. only %d faces was not detected" % (total, not_detect))

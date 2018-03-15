# coding: utf-8

import face_embedding
import argparse
import cv2
import numpy as np
import os, sys, time, datetime

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r18-amf/model,0', help='path to load model.')
parser.add_argument('--ctx', default='cpu,0', type=str, help='context cpu or gpu')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

if __name__ == '__main__':
    model = face_embedding.FaceModel(args)
    img = cv2.imread("./test2.jpg")
    vec = model.get_feature(img, "", "")
    print(type(vec))
    print(vec[0])

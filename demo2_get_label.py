#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import time
timestamp = int(round(time.time()))
import numpy
import deepmodels
import json
import os.path
import argparse
import alignface
import imageutils
import utils
import random


def fit_submanifold_landmarks_to_image(template, original, Xlm, face_d, face_p, landmarks=list(range(68))):
    '''
    Fit the submanifold to the template and take the top-K.

    Xlm is a N x 68 x 2 list of landmarks.
    '''
    lossX = numpy.empty((len(Xlm),), dtype=numpy.float64)
    MX = numpy.empty((len(Xlm), 2, 3), dtype=numpy.float64)
    nfail = 0
    for i in range(len(Xlm)):
        lm = Xlm[i]
        try:
            M, loss = alignface.fit_face_landmarks(Xlm[i], template, landmarks=landmarks, image_dims=original.shape[:2])
            lossX[i] = loss
            MX[i] = M
        except alignface.FitError:
            lossX[i] = float('inf')
            MX[i] = 0
            nfail += 1
    if nfail > 1:
        print('fit submanifold, {} errors.'.format(nfail))
    a = numpy.argsort(lossX)
    return a, lossX, MX


def select(constraints, attributes, filelist):
    LUT = {-1: False, 1: True}

    def admissible(i):
        return LUT[attributes[i]] == constraints

    S = numpy.asarray([i for i in range(len(filelist)) if admissible(i)])
    if len(S) < 1: return []
    return [filelist[i] for i in S]


if __name__ == '__main__':
    random.seed(0)
    # configure by command-line arguments
    parser = argparse.ArgumentParser(description='Generate high resolution face transformations.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='input color image')
    parser.add_argument('--backend', type=str, default='torch', choices=['torch', 'caffe+scipy'], help='reconstruction implementation')
    parser.add_argument('--device_id', type=int, default=0, help='zero-indexed CUDA device')
    parser.add_argument('--K', type=int, default=100, help='number of nearest neighbors')
    parser.add_argument('--scaling', type=str, default='none', choices=['none', 'beta'], help='type of step scaling')
    parser.add_argument('--iter', type=int, default=500, help='number of reconstruction iterations')
    parser.add_argument('--postprocess', type=str, default='mask', help='comma-separated list of postprocessing operations')
    parser.add_argument('--delta', type=str, default='3.5', help='comma-separated list of interpolation steps')
    parser.add_argument('--output_format', type=str, default='png', choices=['png', 'jpg'], help='output image format')
    parser.add_argument('--comment', type=str, default='', help='the comment is appended to the output filename')
    parser.add_argument('--extradata', action='store_true', default=False, help='extra data is saved')
    parser.add_argument('--output', type=str, default='', help='output is written to this pathname')
    parser.add_argument('--include_original', action='store_true', default=False, help='the first column of the output is the original image')
    parser.add_argument('--dataset', type=str, default='facemodel')
    parser.add_argument('--save_vector', action='store_true')
    parser.add_argument('--get_prior', action='store_true')
    parser.add_argument('--vector_path', type=str)
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--load_size', type=int, default=None)

    config = parser.parse_args()
    postprocess = set(config.postprocess.split(','))
    postfix_comment = '_'+config.comment if config.comment else ''
    print(json.dumps(config.__dict__))

    # load models
    minimum_resolution = 200
    if config.backend == 'torch':
        import deepmodels_torch
        model = deepmodels_torch.vgg19g_torch(device_id = config.device_id)
    else:
        raise ValueError('Unknown backend')
    classifier = deepmodels.facemodel_attributes(dataset=config.dataset)
    fields = classifier.fields()
    face_d, face_p = alignface.load_face_detector()

    # Set the free parameters
    K = config.K
    delta_params = [float(x.strip()) for x in config.delta.split(',')]

    X = config.input
    if config.save_vector:
        template, original = alignface.detect_landmarks(X, face_d, face_p)
        image_dims = original.shape[:2]
        if min(image_dims) < minimum_resolution:
            s = float(minimum_resolution) / min(image_dims)
            image_dims = (int(round(image_dims[0] * s)), int(round(image_dims[1] * s)))
            original = imageutils.resize(original, image_dims)
        XF = model.mean_F([original])
        XA = classifier.score([X])[0]

        attributes = classifier._attributes
        filelist = classifier.filelist

        # select positive and negative sets
        P = select(False, attributes, filelist)
        Q = select(True, attributes, filelist)

        # Plm = classifier.lookup_landmarks(P[:4*K])
        # Qlm = classifier.lookup_landmarks(Q[:4*K])
        # idxP, lossP, MP = fit_submanifold_landmarks_to_image(template, original, Plm, face_d, face_p)
        # idxQ, lossQ, MQ = fit_submanifold_landmarks_to_image(template, original, Qlm, face_d, face_p)
        #
        # # Use the K best fitted images
        # xP = [P[i] for i in idxP[:K]]
        # xQ = [Q[i] for i in idxQ[:K]]
        xP = random.sample(P, min(K, len(P)))
        xQ = random.sample(Q, min(K, len(Q)))

        PF = model.mean_F(utils.image_feed(xP[:K], image_dims))
        QF = model.mean_F(utils.image_feed(xQ[:K], image_dims))

        WF = QF - PF
        WF = 1. * WF / numpy.linalg.norm(WF)

        OF = model.mean_F(utils.image_feed([X], image_dims))
        with open(config.vector_path, 'w') as f:
            numpy.savez(f, vector=WF, image_dims=image_dims, origin=OF)
    elif not config.get_prior:
        # load vector
        data = numpy.load(config.vector_path)
        WF = data['vector']
        OF = data['origin']
        image_dims = data['image_dims']
        WF_norm = 1. * WF / numpy.linalg.norm(WF)

        X = [os.path.join(config.dataroot, x.rstrip('\n').strip()) for x in open(X).readlines()]
        X = random.sample(X, min(K, len(X)))
        XF = model.mean_F(utils.image_feed(X, image_dims))
        p = numpy.inner(XF, WF_norm)
        inner_product_path = config.vector_path.replace('.npz', '_inner.npz')
        if os.path.exists(inner_product_path):
            data = numpy.load(inner_product_path)
            plist = list(data['inner_prod'])
            plist.append(p)
        else:
            plist = [p]
        with open(inner_product_path, 'w') as f:
            print(plist)
            numpy.savez(f, inner_prod=plist)
    else:
        X = [os.path.join(config.dataroot, x.rstrip('\n')) for x in open(config.input, 'r').readlines()]

        # load vector
        data = numpy.load(config.vector_path)
        WF = data['vector']
        OF = data['origin']
        image_dims = data['image_dims']
        WF_norm = 1. * WF / numpy.linalg.norm(WF)
        inner_prods = []

        for i in range(len(X)):
            xX = X[i]
            XF = model.mean_F(utils.image_feed([xX], image_dims))

            p = numpy.inner(XF, WF_norm)
            inner_prods.append(p)

        inner_product_path = config.vector_path.replace('.npz', '_prior.npz')
        with open(inner_product_path, 'w') as f:
            print(inner_prods)
            numpy.savez(f, inner_prod=inner_prods)

# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback

import flask

import math
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
import gluoncv as gcv
from gluoncv.data.transforms import image as timage


prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
artifact_file = os.path.join(model_path, 'artifact.json')
model_name = None
with open(artifact_file) as json_file:
    artifact = json.load(json_file)
    model_name = artifact['model_name']
if not model_name:
    raise RuntimeError('Unable to determine saved model name')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            model = gcv.model_zoo.get_model(model_name, pretrained=False, pretrained_base=False)
            model.load_parameters(os.path.join(model_path, 'model.params'), allow_missing=True)
            model.hybridize()
            cls.model = model
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        """
        clf = cls.get_model()
        return clf(input)


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take image data, decode
    image to raw frame, and apply prediction.
    """
    def preprocess(raw_image_buf, short=512, max_size=768,
               mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        orig_image = mx.img.imdecode(raw_image_buf)
        img = timage.resize_short_within(orig_image, short, max_size, mult_base=32)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        return img.expand_dims(0), orig_image

    def filter_results(cids, scores, bboxes, class_names, shape, orig_shape, thresh=0.1):
        scale = float(orig_shape[0]) / shape[0]
        cid, scores, bboxes = [x[0].asnumpy() for x in (cids, scores, bboxes)]
        num = cid.shape[0]
        ret = []
        for i in range(num):
            id = int(cid[i])
            if id < 0:
                break
            if id >= len(class_names):
                continue
            score = float(scores[i])
            if score < thresh:
                continue
            bbox = bboxes[i, :] * scale
            bbox[::2] = np.maximum(0, np.minimum(orig_shape[1], bbox[::2]))
            bbox[1::2] = np.maximum(0, np.minimum(orig_shape[0], bbox[1::2]))
            obj = {
                'id': class_names[id], 'score': score,
                'left': int(bbox[0]), 'top': int(bbox[1]),
                'right': int(bbox[2]), 'bottom': int(bbox[3])}
            ret.append(obj)
        return ret

    data = None

    def parse_default_args(custom_attr):
        thresh = custom_attr.get('threshold', 0.1)
        print('customized attributes parsed: threshold:', thresh)
        short, max_size = 512, 768
        if model_name.startswith('faster_rcnn'):
            short, max_size = 800, 1333
        elif model_name.startswith('ssd_300'):
            short, max_size = 300, 512
        return short, max_size, thresh

    # decode image and apply transformation to image
    custom_attr = flask.request.headers.get('X-Amzn-Sagemaker-Custom-Attributes', {})
    if custom_attr:
        try:
            # expect json attributes
            custom_attr = json.loads(custom_attr)
        except Exception as e:
            raise ValueError('Invalid Custom-Attributes: ' + str(e))
    short, max_size, thresh = parse_default_args(custom_attr)
    if flask.request.content_type in ['image/jpeg', 'image/png']:
        raw = flask.request.data
        data, orig_image = preprocess(raw, short=short, max_size=max_size)
    else:
        return flask.Response(response='This predictor only supports image data, {}'.format(flask.request.content_type), status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    cid, scores, bboxes = ScoringService.predict(data)

    # Convert from output to readable text
    result = json.dumps(filter_results(cid, scores, bboxes, ScoringService.get_model().classes, data.shape[2:], orig_image.shape, thresh))

    return flask.Response(response=result, status=200, mimetype='application/json')

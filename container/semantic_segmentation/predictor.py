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
        output = clf(input)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy().astype('int32')
        return predict


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
    def preprocess(raw_image_buf, size=480,
               mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        orig_image = mx.img.imdecode(raw_image_buf)
        img = timage.imresize(orig_image, size, size)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        return img.expand_dims(0), orig_image

    if flask.request.content_type in ['image/jpeg', 'image/png']:
        raw = flask.request.data
        data, orig_image = preprocess(raw, short=short, max_size=max_size)
    else:
        return flask.Response(response='This predictor only supports image data, {}'.format(flask.request.content_type), status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    pred = ScoringService.predict(data)

    # Convert from output to readable text
    result = json.dumps({'prediction': pred.tolist()})

    return flask.Response(response=result, status=200, mimetype='application/json')

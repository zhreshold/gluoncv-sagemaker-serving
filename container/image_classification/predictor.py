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
from gluoncv.data import ImageNet1kAttr


prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
artifact_file = os.path.join(model_path, 'artifact.json')
mode_name = None
preprocessor = None
classes = ImageNet1kAttr().classes
with open(artifact_file) as json_file:
    artifact = json.load(json_file)
    model_name = artifact['model_name']
    if model_name.startswith('inception') or model_name.startswith('googlenet'):
        input_size = 299
    elif model_name == 'resnest101':
        input_size = 256
    elif model_name == 'resnest200':
        input_size = 320
    elif model_name == 'resnest269':
        input_size = 416
    else:
        input_size = 224
    resize = int(math.ceil(input_size/0.875))
    preprocessor = transforms.Compose([
        transforms.Resize(resize, keep_ratio=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
if not model_name or not preprocessor:
    raise RuntimeError('Unable to determine saved model name')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = {}                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls, feature=''):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model.get(feature, None) == None:
            net = gcv.model_zoo.get_model(model_name)
            net.load_parameters(os.path.join(model_path, 'model.params'))
            if feature == '':
                model = net
            else:
                if feature not in list(net._children):
                    raise ValueError('Layer: {} does not exist in network, available choices are: \n' +
                        '\n'.join(list(net._children)))
                model = mx.gluon.nn.HybridSequential(prefix=feature)
                for child in list(net._children):
                    model.add(getattr(net, child))
                    if child == feature:
                        break
            model.hybridize(static_alloc=True, static_shape=True)
            cls.model[feature] = model
        return cls.model[feature]

    @classmethod
    def predict(cls, input, feature=''):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model(feature)
        if feature:
            print('Customized feature extraction:', feature)
            return np.array2string(clf(input).asnumpy(), separator=',', threshold=np.inf)
        pred = clf(input)
        topk = 5
        ind = nd.topk(pred, k=topk)[0].astype('int')
        scores = nd.softmax(pred)
        responses = []
        for i in range(topk):
            responses.append('[%s], with probability %.3f.\n'%
                  (classes[ind[i].asscalar()], scores[0][ind[i]].asscalar()))
        return ''.join(responses)


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
    data = None

    # Do imdecoding and preprocessing
    if flask.request.content_type in ('image/jpeg', 'image/png'):
        raw_data = np.fromstring(flask.request.data, np.uint8)
        img = mx.image.imdecode(raw_data)
        data = preprocessor(img).expand_dims(0)
    else:
        return flask.Response(response='This predictor only supports image/jpeg or image/png', status=415, mimetype='text/plain')

    custom_attr = flask.request.headers.get('X-Amzn-Sagemaker-Custom-Attributes', {})
    if custom_attr:
        try:
            # expect json attributes
            custom_attr = json.loads(custom_attr)
        except Exception as e:
            raise ValueError('Invalid Custom-Attributes: ' + str(e))
    print('Invoked with {} records, custom attributes: {}'.format(data.shape[0], custom_attr))
    feature = custom_attr.get('feature', '')

    # Do the prediction
    result = ScoringService.predict(data, feature)

    return flask.Response(response=result, status=200, mimetype='text/plain')

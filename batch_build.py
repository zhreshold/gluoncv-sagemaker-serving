import subprocess
import sys
import os

import gluoncv as gcv

def build(app):
    try:
        print('Building for {}...'.format(app))
        subprocess.check_output(['cd container && ./build_and_push.sh gluoncv-{} {}'.format(app.replace('_', '-'), app)],
                                 shell=True)
    except subprocess.CalledProcessError as e:
        out_bytes = e.output       # Output generated before error
        code      = e.returncode   # Return code
        print(out_bytes)
        sys.exit(code)

def build_image_classification(models):
    build('image_classification')
    for model in models:
        print(model, '...')

if __name__ == '__main__':
    image_classification_models = ['resnet50_v1b', 'mobilenetv3_large']
    build_image_classification(image_classification_models)

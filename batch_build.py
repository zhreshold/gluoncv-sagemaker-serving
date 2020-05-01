import subprocess
import sys

import gluoncv as gcv

def build_image_classification(models):
    for model in models:
        try:
            print('Building for {}...'.format(model))
            subprocess.check_output(['container/build_and_push.sh',
                                     'gluoncv-{}'.format(model),
                                     'image_classification'])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output       # Output generated before error
            code      = e.returncode   # Return code
            print(out_bytes)
            sys.exit(code)

if __name__ == '__main__':
    image_classification_models = ['resnet50_v1b', 'mobilenetv3_large']
    build_image_classification(image_classification_models)

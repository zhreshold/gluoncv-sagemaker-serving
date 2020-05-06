# gluoncv-sagemaker
SageMaker serving for GluonCV models

# Pre-requisite

- SageMaker notebook(recommended)
- Computer with AWS CLI configured.

# Usage:

The following instructions works for `SageMaker` notebook instance

- First activate a `mxnet_p36` env by `source activate mxnet_p36` in a new terminal.
- Then install required python packages: `pip install -r requirements.txt`
- Execute the batch building script: `python batch_build.py`.

The script will build `ModelPackage` for models listed in `image_classification.txt` and `object_detection.txt`. Due to the validation process, it might take very long time to finish.

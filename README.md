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


# Marketplace listing Metadata Generator
```bash
python batch_describe.py
```

The sample output looks like: https://gist.github.com/zhreshold/f616f5b894d386701b4c85a4b40d200c

## TODO
- [ ] Automatic ARN injection for built ModelPackage
- [ ] Object Detection metadata
- [ ] Semantic Segmentation metadata

import subprocess
import sys
import os
import argparse
import json
import time

from sagemaker import get_execution_role
import sagemaker as sage
import boto3
from src.inference_specification import InferenceSpecification
from src.modelpackage_validation_specification import ModelPackageValidationSpecification

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
        build_image_classification_impl(model)

def build_image_classification_impl(model_name, deploy_test=False, region='us-west-2'):
    # role
    common_prefix = "DEMO-gluoncv-model-zoo"
    training_input_prefix = common_prefix + "/training-input-data"
    role = get_execution_role()
    sess = sage.Session()

    # create estimator
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    image = '{}.dkr.ecr.{}.amazonaws.com/gluoncv-image-classification:latest'.format(account, region)
    TRAINING_WORKDIR = "data/training"

    training_input = sess.upload_data(TRAINING_WORKDIR, key_prefix=training_input_prefix)
    print ("Training Data Location " + training_input)
    classifier = sage.estimator.Estimator(image,
                           role, 1, 'ml.c4.2xlarge',
                           output_path="s3://{}/output".format(sess.default_bucket()),
                           sagemaker_session=sess,
                           hyperparameters={'model_name': model_name})
    classifier.fit(training_input)

    TRANSFORM_WORKDIR = "data/transform"
    batch_inference_input_prefix = common_prefix + "/batch-inference-input-data"
    transform_input = sess.upload_data(TRANSFORM_WORKDIR, key_prefix=batch_inference_input_prefix) + "/cat1.jpg"

    # deploy
    if deploy_test:
        model = classifier.create_model()
        predictor = classifier.deploy(1, 'ml.m4.xlarge')
        with open('data/transform/cat1.jpg', 'rb') as f:
            x = f.read()
            print(predictor.predict(x, initial_args={'ContentType':'image/jpeg'}).decode('utf-8'))
        sess.delete_endpoint(predictor.endpoint)

    smmp = boto3.client('sagemaker', region_name=region, endpoint_url="https://sagemaker.{}.amazonaws.com".format(region))
    modelpackage_inference_specification = InferenceSpecification().get_inference_specification_dict(
        ecr_image=image,
        supports_gpu=True,
        supported_content_types=["image/jpeg", "image/png"],
        supported_mime_types=["text/plain"])

    # Specify the model data resulting from the previously completed training job
    modelpackage_inference_specification["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]=classifier.model_data

    # validation specificiation
    modelpackage_validation_specification = ModelPackageValidationSpecification().get_validation_specification_dict(
        validation_role = role,
        batch_transform_input = transform_input,
        content_type = "image/jpeg",
        instance_type = "ml.c4.xlarge",
        output_s3_location = 's3://{}/{}'.format(sess.default_bucket(), common_prefix))

    model_package_name = "gluoncv-{}".format(model_name.replace('_', '-')) + str(round(time.time()))
    create_model_package_input_dict = {
        "ModelPackageName" : model_package_name,
        "ModelPackageDescription" : "Model to perform image classification or extract image features by deep learning",
        "CertifyForMarketplace" : True
    }
    create_model_package_input_dict.update(modelpackage_inference_specification)
    create_model_package_input_dict.update(modelpackage_validation_specification)

    smmp.create_model_package(**create_model_package_input_dict)

    while True:
        response = smmp.describe_model_package(ModelPackageName=model_package_name)
        status = response["ModelPackageStatus"]
        print (model_name, ':', status)
        if (status == "Completed" or status == "Failed"):
            print (response["ModelPackageStatusDetails"])
            break
        time.sleep(5)

if __name__ == '__main__':
    image_classification_models = ['resnet50_v1b', 'mobilenetv3_large']
    build_image_classification(image_classification_models)

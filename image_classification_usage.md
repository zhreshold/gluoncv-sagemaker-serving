### Input
Supported content types: `image/jpeg`, `image/png`

### Output
Content type: `application/json`

Sample output for top-5 prediction:
```json
{
    "prediction": [
        {
            "class": "lynx",
            "prob": 0.253
        },
        {
            "class": "Egyptian cat",
            "prob": 0.252
        },
        {
            "class": "tiger cat",
            "prob": 0.106
        },
        {
            "class": "tabby",
            "prob": 0.063
        },
        {
            "class": "soft-coated wheaten terrier",
            "prob": 0.041
        }
    ]
}
```

Sample output for `flat` layer feature extraction:
(Note that you can extract features from different layers, but the name of layers can vary, please refer to
  detailed information from output)
```json
{
    "feature": [[6.44997060e-01,2.90736866e+00,4.06486607e+00,3.07884037e-01,
      1.50389830e-02,3.26474607e-01,9.00917351e-01,2.11797309e+00,
      ...
      2.86869437e-01,3.44580293e-01,5.72227407e-03,1.88450420e+00,
      2.56996118e-02,9.89234626e-01,7.65379488e-01,2.22946095e+00]]
}
```

## Invoking endpoint

### AWS CLI Command
You can invoke endpoint using AWS CLI:
```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name "endpoint-name" –body fileb://input.jpg --content-type image/jpeg --accept application/json out.json
```

You can specify the name of layer you want feature to be extracted from
```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name "endpoint-name" –body fileb://input.jpg --content-type image/jpeg --accept application/json --custom-attributes '{"feature": "flat"}' out.json
```

Substitute the following parameters:
* `"endpoint-name"` - name of the inference endpoint where the model is deployed
* `input.jpg` - input image to do the inference on
* `image/jpeg` - MIME type of the given input image (above)
* `--custom-attributes` - If specified, the network won't perform prediction but rather extract the feature
* `out.json` - filename where the inference results are written to

### Python
Real-time inference snippet (more detailed example can be found in sample notebook):
```python
runtime = boto3.Session().client(service_name='runtime.sagemaker')
bytes_image = open('input.jpg', 'rb').read()
response = runtime.invoke_endpoint(EndpointName='endpoint-name', ContentType='image/jpeg', Body=bytes_image)
# or extract feature
# response = runtime.invoke_endpoint(EndpointName='endpoint-name', ContentType='image/jpeg', Body=bytes_image, CustomAttributes=custom_attributes)
response = response['Body'].read()
results = json.loads(response)
```

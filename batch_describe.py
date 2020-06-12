"""Descript the listing information for each model"""
import json

VERSION = 1.0
RELEASE_NOTES = 'Pretrained with GluonCV 0.7.0'


class ListingMeta(dict):
    def __init__(self, title, short_desr, overview,
                 cat1, cat2, hl1, hl2,
                 keywords, lnk1_text, lnk1_url,
                 usage, realtime_inst_type):
        self.title = title
        self._meta = {title: [
            self._ff(1, 'Product title', 'String', title, 40),
            self._ff(2, 'Short product description', 'String', short_desr, 140),
            self._ff(3, 'Product Overview', 'String', overview, 600),
            self._ff(4, 'Product category 1', 'Category', cat1),
            self._ff(5, 'Product category 2', 'Category', cat2),
            self._ff(7, 'Highlight 1', 'String', hl1, 600),
            self._ff(8, 'Highlight 2', 'String', hl2, 600),
            self._ff(10, 'Product Logo', 'File', 'mxnet_logo.png'),
            self._ff(11, 'Search keywords', 'String', keywords),
            self._ff(12, 'Resource link 1 - text', 'String', lnk1_text),
            self._ff(13, 'Resource link 1 - URL', 'String', lnk1_url),
            self._ff(18, 'Are you offering support for this product?', 'Bool', 'Yes'),
            self._ff(19, 'Support information or contact information', 'String', 'Model reference: https://gluon-cv.mxnet.io/model_zoo/classification.html . Model supported is available from GluonCV. Search for questions and open new issues to ask questions.'),
            self._ff(20, 'URL to support resources', 'String', 'https://gluon-cv.mxnet.io/index.html'),

            # per version
            # self._ff(22, 'Model Package ARN', 'String', ''),  (TODO)
            self._ff(23, 'Customer facing version number', 'String', str(VERSION)),
            self._ff(24, 'General usage information', 'String', '', 2000),
            self._ff(25, 'Release Notes', 'String', str(RELEASE_NOTES), 600),
            self._ff(26, 'Set recommended batch transform instance type', 'MLInstanceType', 'ml.c4.xlarge'),
            self._ff(27, 'Set recommended realtime inference instance type', 'MLInstanceType', realtime_inst_type),

        ]}
        dict.__init__(self, self._meta)

    def _ff(self, index, name, type, value, max_char=None):
        if type == 'String' and max_char is not None:
            assert len(value) <= int(max_char), '{}: "{}" length {} exceeds {}'.format(name, value, len(value), max_char)
        field = {
            '#': index, 'Field Name': name, 'Type': type, 'Value': value,
        }
        return field


def describe_classification(lst):
    with open(lst, 'rt') as f:
        models = [m.strip() for m in f.readlines()]
        meta = []
    for model in models:
        title = 'GluonCV {} classifier'.format(model)
        short_desr = 'Image feature extraction and ImageNet category prediction using {}, provided by GluonCV'.format(model)
        overview = 'This model provides intermediate image feature extraction functionality for image classification. It can also provide top-5 category predictions out of 1000 classes on ImageNet. It also provides high-quality features as well.'
        cat1 = "Machine Learning > Computer Vision"
        cat2 = "Machine Learning > Image > Classification-Image"
        hl1 = "This model can extract high-quality image features efficiently."
        hl2 = "This model can also predict top-5 predictions on ImageNet."
        keywords = "imagenet, image classification, {}".format(model)
        lnk1_text = "GluonCV website"
        lnk1_url = "https://gluon-cv.mxnet.io/"
        usage = open('image_classification_usage.md', 'rt').read()
        realtime_inst_type='ml.c4.xlarge'
        metadata = ListingMeta(title=title, short_desr=short_desr, overview=overview,
                               cat1=cat1, cat2=cat2, hl1=hl1, hl2=hl2, keywords=keywords,
                               lnk1_text=lnk1_text, lnk1_url=lnk1_url, usage=usage,
                               realtime_inst_type=realtime_inst_type)
        meta.append(metadata)
    return meta

if __name__ == '__main__':
    meta = describe_classification('image_classification.txt')
    print(json.dumps(meta))

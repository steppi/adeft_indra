import os
import json
import boto3
import tempfile

from adeft.download import get_s3_models
from adeft.util import str2filename

from adeft_indra.locations import S3_BUCKET_ADEFT, S3_MODELS_PATH


def model_to_s3(disambiguator):
    grounding_dict = disambiguator.grounding_dict
    names = disambiguator.names
    classifier = disambiguator.classifier

    shortforms = disambiguator.shortforms
    model_name = '&'.join(sorted(str2filename(shortform)
                                 for shortform in shortforms))
    model_map = {key: model_name for key in grounding_dict}
    s3_models = get_s3_models()
    s3_models.update(model_map)
    client = boto3.client('s3')
    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            json.dump(s3_models, f)
        client.upload_file(temp.name, S3_BUCKET_ADEFT,
                           os.path.join(S3_MODELS_PATH, 's3_models.json'))

    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            json.dump(grounding_dict, f)
        client.upload_file(temp.name, S3_BUCKET_ADEFT,
                           os.path.join(S3_MODELS_PATH, model_name,
                                        f'{model_name}_grounding_dict.json'))

    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            json.dump(names, f)
        client.upload_file(temp.name, S3_BUCKET_ADEFT,
                           os.path.join(S3_MODELS_PATH, model_name,
                                        f'{model_name}_names.json'))

    with tempfile.NamedTemporaryFile() as temp:
        classifier.dump_model(temp.name)
        client.upload_file(temp.name, S3_BUCKET_ADEFT,
                           os.path.join(S3_MODELS_PATH, model_name,
                                        f'{model_name}_model.gz'))

import argparse
# import base64
import json

import googleapiclient.discovery
import six
from dotenv import load_dotenv
import os


def get_credential():
    load_dotenv()
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print('you must set GOOGLE_APPLICATION_CREDENTIALS in .env file')
        return False
    else:
        return True


def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.
    """
    # Create the AI Platform service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def main(input_json_path, project, model, version=None):
    """Send user input to the prediction service."""
    if not get_credential():
        return None

    # load input with json
    with open(input_json_path, 'r') as f:
        user_input = json.load(f)

    try:
        # predict
        result = predict_json(
            project, model, user_input, version=version)
    except RuntimeError as err:
        print(str(err))
    else:
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json_path',
        help='Json path for request input',
        type=str,
        required=True
    )
    parser.add_argument(
        '--project',
        help='Project in which the model is deployed',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model',
        help='Model name',
        type=str,
        required=True
    )
    parser.add_argument(
        '--version',
        help='Name of the version.',
        type=str
    )
    args = parser.parse_args()
    main(
        args.json_path,
        args.project,
        args.model,
        version=args.version,
    )

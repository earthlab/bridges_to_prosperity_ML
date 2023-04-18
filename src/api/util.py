import os
import yaml

from definitions import SECRETS_FILE_PATH


def generate_secrets_file(nasaearthdata_u: str = '', nasaearthdata_p: str = '', openstreetmap_u: str = '',
                          openstreetmap_p: str = ''):
    fields = {
        'nasaearthdata': {
            'username': nasaearthdata_u,
            'password': nasaearthdata_p
        },
        'openstreetmap': {
            'username': openstreetmap_u,
            'password': openstreetmap_p
        }
    }
    with open(SECRETS_FILE_PATH, 'w+') as f:
        yaml.dump(fields, f)

    print(f'Secrets file written to {SECRETS_FILE_PATH} with credentials')

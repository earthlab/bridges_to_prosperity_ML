import getpass
import os
from typing import Tuple, List
from argparse import Namespace
from tqdm import tqdm
import multiprocessing as mp

import yaml

from definitions import B2P_DIR, REGION_FILE_PATH
from src.api.util import generate_secrets_file

import requests


BASE_URL = 'https://appeears.earthdatacloud.nasa.gov/api/'


def _download_task(args: Namespace):
    url = os.path.join(BASE_URL, 'bundle', args.task_id, args.file_id)

    header = {
        'Authorization': f'Bearer {args.auth_token}'
    }

    response = requests.get(url, headers=header, allow_redirects=True, stream=True)

    os.makedirs(args.out_dir, exist_ok=True)

    print(args.file_name)
    progress_bar = tqdm(total=args.file_size, unit='iB', unit_scale=True)
    with open(os.path.join(args.out_dir, args.file_name), 'wb') as f:
        for data in response.iter_content(chunk_size=8192):
            progress_bar.update(len(data))
            f.write(data)


class BaseAPI:
    """
    Defines all the attributes and methods common to the child APIs.
    """

    BASE_URL = BASE_URL

    def __init__(self):
        """
        Initializes the common attributes required for each data type's API
        """
        self._username, self._password = self._get_auth_credentials()
        self._auth_token = self._get_auth_token()
        self._core_count = os.cpu_count()

        self._AUTH_HEADER = {
            'Authorization': f'Bearer {self._auth_token}'
        }

        self._session_tasks = []

    def _get_auth_token(self) -> str:
        url = os.path.join(self.BASE_URL, 'login')

        response = requests.post(url, auth=(self._username, self._password))

        if response.status_code == 200:
            return response.json()['token']
        else:
            raise ValueError(response.json())

    @property
    def session_tasks(self):
        return self._session_tasks

    @staticmethod
    def _get_auth_credentials() -> Tuple[str, str]:
        """
        Ask the user for their urs.earthdata.nasa.gov username and login
        Returns:
            username (str): urs.earthdata.nasa.gov username
            password (str): urs.earthdata.nasa.gov password
        """
        secrets_file_path = os.path.join(B2P_DIR, 'secrets.yaml')
        if not os.path.exists(secrets_file_path):
            print('Please input your earthdata.nasa.gov username and password. If you do not have one, you can register'
                  ' here: https://urs.earthdata.nasa.gov/users/new')
            username = input('Username:')
            password = getpass.getpass('Password:', stream=None)
            generate_secrets_file(nasaearthdata_u=username, nasaearthdata_p=password)

        with open(secrets_file_path, 'r') as f:
            secrets = yaml.load(f, Loader=yaml.FullLoader)

        username = secrets['nasaearthdata']['username']
        password = secrets['nasaearthdata']['password']

        if username == '' or password == '':
            username = input('Username:')
            password = getpass.getpass('Password:', stream=None)
            secrets['nasaearthdata']['username'] = username
            secrets['nasaearthdata']['password'] = password
            with open(secrets_file_path, 'w') as f:
                yaml.dump(secrets, f)

        return username, password

    def _authorized_get_request(self, url):
        response = requests.get(url, headers=self._AUTH_HEADER)
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(response.json())

    def list_task_descriptions(self):
        url = os.path.join(self.BASE_URL, 'task')
        return self._authorized_get_request(url)

    def get_task_description(self, task_id: str):
        url = os.path.join(self.BASE_URL, 'task', task_id)
        return self._authorized_get_request(url)

    def list_task_statuses(self):
        url = os.path.join(self.BASE_URL, 'status')
        return self._authorized_get_request(url)

    def get_task_status(self, task_id: str):
        url = os.path.join(self.BASE_URL, 'status', task_id)
        return self._authorized_get_request(url)

    def list_task_files(self, task_id: str):
        url = os.path.join(self.BASE_URL, 'bundle', task_id)
        return self._authorized_get_request(url)

    def download_file(self, task_id: str, file_id: str, out_dir: str):
        url = os.path.join(self.BASE_URL, 'bundle', task_id, file_id)

        response = requests.get(url, headers=self._AUTH_HEADER, allow_redirects=True, stream=True)

        file_response = self.list_task_files(task_id)
        file_name = None
        file_size = None
        for file in file_response['files']:
            if file['file_id'] == file_id:
                file_name = os.path.basename(file['file_name'])
                file_size = file['file_size']
                break
        if file_size is None or file_name is None:
            raise LookupError('Could not find file id in manifest')

        if response.status_code == 200:
            os.makedirs(out_dir, exist_ok=True)

            progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
            with open(os.path.join(out_dir, file_name), 'wb') as f:
                for data in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(data))
                    f.write(data)
        else:
            print(response.json())

    def list_session_tasks_status(self):
        for task in self.session_tasks:
            print(task['task_id'])
            print(self.get_task_status(task))


class Elevation(BaseAPI):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _bbox_to_polygon(bbox: List[float]):
        return [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]

    def download_session_task_files(self, out_dir: str):
        args = []
        for task in self.session_tasks:
            task_id = task['task_id']
            status_response = self.get_task_status(task_id)
            for session_task in self._session_tasks:
                if session_task['task_id'] == task_id:
                    session_task['status'] = status_response['status']

            if 'status' in status_response and status_response['status'] == 'done':
                files_response = self.list_task_files(task_id)
                for file_info in files_response['files']:
                    file_name = os.path.basename(file_info['file_name'])
                    file_id = file_info['file_id']
                    file_size = file_info['file_size']

                    if 'NASADEM_NC' in file_name and '_HGT_' in file_name and (
                            '.tif' in file_name or '.ncdf4' in file_name):
                        args.append(Namespace(task_id=task_id, file_id=file_id, file_name=file_name,
                                              file_size=file_size, out_dir=out_dir, auth_token=self._auth_token))

        for arg in args:
            print(vars(arg))
            _download_task(arg)
        print(f'Session tasks: {self.session_tasks}')
        # with mp.Pool(mp.cpu_count() - 1) as pool:
        #     for _ in tqdm(pool.imap_unordered(_download_task, args), total=len(args)):
        #         pass

    def submit_task(self, region: str = None, district: str = None, bbox: List[float] = None,
                    start_date: str = '02-11-2000', end_date: str = '02-21-2000', file_type: str = 'geotiff',
                    projection: str = "native"):
        if region is None and bbox is None:
            raise ValueError('Must specify region or bbox')

        if district is not None and region is None:
            raise ValueError('Must specify a region with the district')

        url = os.path.join(self.BASE_URL, 'task')

        header = self._AUTH_HEADER
        header['Content-Type'] = 'application/json'

        if region is not None:
            with open(REGION_FILE_PATH, 'r') as f:
                region_file_info = yaml.load(f, Loader=yaml.FullLoader)
        region_info = region_file_info[region]

        polygons = []
        if district is not None:
            polygons.append(self._bbox_to_polygon(region_info['districts'][district]['bbox']))
        else:
            for district in region_info['districts']:
                polygons.append(self._bbox_to_polygon(region_info[district]['bbox']))

        body = {
            "task_type": "area",
            "task_name": f"elevation_{1}",
            "params":
                {
                    "dates": [{
                        "startDate": start_date,
                        "endDate": end_date
                    }],
                    "layers": [{
                        "product": "NASADEM_NC.001",
                        "layer": "NASADEM_HGT"
                    }],
                    "geo": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": polygons
                                },
                                "type": "Feature", "properties": {}
                            }
                        ]
                    },
                    "output":
                        {
                            "format":
                                {
                                    "type": file_type
                                },
                            "projection": projection
                        }
                }
        }

        response = requests.post(url, headers=header, json=body)

        if response.status_code == 202:
            self._session_tasks.append(response.json())
            print(response.json())
        else:
            raise ValueError(response.json())


    # def _configure(self) -> None:
    #     """
    #     Queries the user for credentials and configures SSL certificates
    #     """
    #     if self._username is None or self._password is None:
    #         username, password = self._cred_query()
    #
    #         self._username = username
    #         self._password = password
    #
    #     # This is a macOS thing... need to find path to SSL certificates and set the following environment variables
    #     ssl_cert_path = certifi.where()
    #     if 'SSL_CERT_FILE' not in os.environ or os.environ['SSL_CERT_FILE'] != ssl_cert_path:
    #         os.environ['SSL_CERT_FILE'] = ssl_cert_path
    #
    #     if'REQUESTS_CA_BUNDLE' not in os.environ or os.environ['REQUESTS_CA_BUNDLE'] != ssl_cert_path:
    #         os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path

import os
import requests
import zipfile

from tqdm import tqdm


"""
Adapted from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
"""

def download_file_from_google_drive(id, destination):
  URL = "https://docs.google.com/uc?export=download"
  session = requests.Session()

  response = session.get(URL, params={ 'id': id }, stream=True)
  token = get_confirm_token(response)

  if token:
    params = { 'id' : id, 'confirm' : token }
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, destination)

def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None

def save_response_content(response, destination, chunk_size=32*1024):
  total_size = int(response.headers.get('content-length', 0))
  with open(destination, "wb") as f:
    for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
              unit='B', unit_scale=True, desc=destination):
      if chunk: # filter out keep-alive new chunks
        f.write(chunk)

def unzip(filepath):
  print("Extracting: " + filepath)
  dirpath = os.path.dirname(filepath)
  with zipfile.ZipFile(filepath) as zf:
    zf.extractall(dirpath)
  os.remove(filepath)

def download_celeb_a(dirpath):
  data_dir = 'img_align_celebA'
  if os.path.exists(os.path.join(dirpath, data_dir)):
    print('Found Celeb-A - skip')
    return
  else:
    os.makedirs(dirpath)

  img_zip_gdrive_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
  list_attr_gdrive_id = "0B7EVK8r0v71pblRyaVFSWGxPY0U"
  eval_partition_gdrive_id = "0B7EVK8r0v71pY0NSMzRuSXJEVkk"

  list_attr_path = os.path.join(dirpath, "list_attr_celeba.txt")
  eval_partition_path = os.path.join(dirpath, "list_eval_partition.txt")
  img_align_zip_path = os.path.join(dirpath, "img_align_celeba.zip")

  download_file_from_google_drive(list_attr_gdrive_id, list_attr_path)
  download_file_from_google_drive(eval_partition_gdrive_id, eval_partition_path)
  download_file_from_google_drive(img_zip_gdrive_id, img_align_zip_path)

  zip_dir = ''
  with zipfile.ZipFile(img_align_zip_path) as zf:
    zip_dir = zf.namelist()[0]
    zf.extractall(dirpath)
  os.remove(img_align_zip_path)
  os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

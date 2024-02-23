import json
import os
import pickle

import boto3
import datasets
import tqdm

from icl.constants import BIGRAMS_FILEPATH
from icl.monitoring import stdlogger

AWS_LANGUAGE_BUCKET_NAME = os.getenv('AWS_LANGUAGE_BUCKET_NAME', os.getenv('AWS_BUCKET_NAME'))

def get_bigrams(file_path=BIGRAMS_FILEPATH)    
    if not os.path.exists(file_path): 
        client = boto3.client('s3')
        client.download_file(AWS_LANGUAGE_BUCKET_NAME, 'other/language/bigram_freq_percents.pkl', file_path)
    
    with open(file_path, 'rb') as file:
        return pickle.load(file)
import json
import os
import pickle

import boto3
import datasets
import numpy as np
import tqdm

from icl.constants import LANGUAGE_FILEPATH, UNIGRAMS_FILEPATH
from icl.language.data import gen_samples
from icl.language.utils import save_to_bucket, translate_int_to_str
from icl.monitoring import stdlogger


def display_unigrams(unigrams, model, k=1000):
    print(f"----- First {k} unigrams -----")

    for i, unigram in enumerate(unigrams):
        if i >= k:
            break

        print(f'{translate_int_to_str(i, model)}: {unigram}')


def gen_unigrams(model, file_path=LANGUAGE_FILEPATH, num_lines=5_000_000, start=0, verbose=True):
    for row, tokens in enumerate(gen_samples(model, file_path, num_lines=num_lines, start=start, verbose=verbose)):
        for i in range(len(tokens) - 2):
            yield row, tokens[i]


def compute_unigram_freqs(model, file_path=LANGUAGE_FILEPATH, num_lines=5_000_000, start=0, verbose=True, vocab_size=5_000):
    unigrams = np.zeros(vocab_size)

    print("Computing unigram frequencies...")
    last_row = 0

    for i, (row, unigram) in enumerate(gen_unigrams(model, file_path, num_lines=num_lines, start=start, verbose=verbose)):
        unigrams[unigram] += 1

        if row > 0 and row % 50_000 == 0 and row != last_row and verbose:
            print(f"------- {row} rows -------")
            display_unigrams(unigrams / i, model)
            last_row = row

    return unigrams / i


def get_unigrams(file_path=UNIGRAMS_FILEPATH):   
    if not os.path.exists(file_path): 
        client = boto3.client('s3')
        client.download_file('devinterp', 'other/language/unigram_freq_percents.pkl', file_path)
    
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    

if __name__ == "__main__":
    from icl.language.model import get_model
    model = get_model()
    unigrams = compute_unigram_freqs(model)

    print("Final probabilities")
    display_unigrams(unigrams, model, k=5000)
    
    with open(UNIGRAMS_FILEPATH, 'wb') as file:
        pickle.dump(unigrams, file)
    
    save_to_bucket(UNIGRAMS_FILEPATH, unigrams)
from typing import Callable, Literal, Optional, Union

import typer

from icl.constants import LANGUAGE_FILEPATH
from icl.language.bigrams import get_bigrams
from icl.language.data import gen_samples
from icl.language.model import get_model
from icl.language.utils import save_to_bucket, translate_int_to_str

app = typer.Typer()

def preview_trigrams_dict(trigrams_dict, model):
    print("----- Top 1000 trigrams -----")
    for k, v in list(trigrams_dict.items())[:1000]:
        trigram = translate_int_to_str(k, model)
        print(f'{trigram}: {v}')


def gen_trigrams(model, file_path=LANGUAGE_FILEPATH, num_lines=5_000_000, start=0, verbose=True):
    for row, tokens in enumerate(gen_samples(model, file_path, num_lines=num_lines, start=start, verbose=verbose)):
        for i in range(len(tokens) - 2):
            yield row, tuple(tokens[i:i+3])


def gen_skip_trigrams(model, file_path=LANGUAGE_FILEPATH, num_lines=5_000_000, start=0, min_skip=10, max_skip=20, verbose=True):
    for row, tokens in enumerate(gen_samples(model, file_path, num_lines=num_lines, start=start, verbose=verbose)):
        for i in range(len(tokens) - 2):
            for skip in range(min_skip, max_skip):
                if i + skip + 1 < len(tokens):
                    yield row, (tokens[i], tokens[i+skip], tokens[i+skip+1])

   
def get_top_k_trigrams(
        model,
        file_path=LANGUAGE_FILEPATH, 
        k=10, 
        multiplier=2, 
        padding=10, 
        start=0,
        num_lines=5_000_000, 
        verbose=True,
        save_fn=None,
        save_every=10_000,
        clean_fn = None
):
    def _clean(table, k=k, padding=padding):
        return dict(sorted(table.items(), key=lambda x: x[1], reverse=True)[:k + padding])
    
    clean_fn = clean_fn or _clean
    table = {}

    for row, trigram in gen_trigrams(model, file_path, start=start, num_lines=num_lines, verbose=verbose):
        table[trigram] = table.get(trigram, 0) + 1
        
        if len(table) > multiplier * k:
            table = clean_fn(table)

            if verbose:
                print("----- Top 1000 trigrams -----")
                preview_trigrams_dict(table, model)

        if save_fn is not None and row % save_every == 0:
            save_fn(row, clean_fn(table))

    return clean_fn(table)


def get_top_k_skip_trigrams(
        model,
        file_path=LANGUAGE_FILEPATH, 
        k=10, 
        multiplier=2, 
        padding=10, 
        start=0,
        num_lines=5_000_000, 
        min_skip=10,
        max_skip=20,
        verbose=True,
        save_fn=None,
        save_every=10_000,
        clean_fn = None
):
    def _clean(table, k=k, padding=padding):
        return dict(sorted(table.items(), key=lambda x: x[1], reverse=True)[:k + padding])
    
    clean_fn = clean_fn or _clean

    table = {}

    for row, trigram in gen_skip_trigrams(model, file_path, start=start, num_lines=num_lines, min_skip=min_skip, max_skip=max_skip, verbose=verbose):
        table[trigram] = table.get(trigram, 0) + 1
        
        if len(table) > multiplier * k:
            table = clean_fn(table)

            if verbose:
                print("----- Top 1000 trigrams -----")
                preview_trigrams_dict(table, model)

        if save_fn is not None and row % save_every == 0:
            save_fn(row, clean_fn(table))

    return clean_fn(table)

@app.command()
def main(
    start: int = typer.Option(0, help="Start row"),
    num_lines: int = typer.Option(5_000_000, help="Number of rows to process"),
    k: int = typer.Option(1_000_000, help="Top k"),
    multiplier: int = typer.Option(10, help="Multiplier"),
    padding: int = typer.Option(1_000_000, help="Padding"),
    skip: bool = typer.Option(False, help="Skip trigrams"),
    min_skip: int = typer.Option(10, help="Minimum skip"),
    max_skip: int = typer.Option(20, help="Maximum skip"),
    verbose: bool = typer.Option(True, help="Verbose"),
    save_every: int = typer.Option(10_000, help="Save every"),
    process: Optional[Union[None, Literal['div_bigram', 'div_bigram_x_unigram']]] = typer.Option(None, help="Process variant"),
):
    def _save_fn(row, table):
        save_to_bucket(table, f"trigrams-{start}-to-{start+num_lines}{'-' + process if process is not None else ''}/{row+1}.pkl")

    vocab_size = 5_000
    bigram_freqs = get_bigrams()
    
    def divide_by_last_2_bigram(trigram):
        # print(translate_int_to_str(trigram), bigram_freqs[trigram[1], trigram[2]])
        return 1. / (vocab_size * bigram_freqs[trigram[1], trigram[2]])

    model = get_model(num_layers=2)

    if skip:
        get_top_k_skip_trigrams(
            model, 
            k=k, 
            multiplier=multiplier, 
            padding=padding, 
            start=start, 
            num_lines=num_lines, 
            min_skip=min_skip, 
            max_skip=max_skip, 
            verbose=verbose,
            save_fn=_save_fn,
            save_every=save_every
        )

    else:   
        get_top_k_trigrams(
            model, 
            k=k, 
            multiplier=multiplier, 
            padding=padding, 
            start=start, 
            num_lines=num_lines, 
            verbose=verbose,
            save_fn=_save_fn,
            save_every=save_every
        )


if __name__ == "__main__":
    app()

def zero_ablated(model, include=['token_embedding', 'postn_embedding', 'unembedding.0', 'unembedding.1']):
    from icl.regression.model import (from_predicted_token_sequence,
                                      to_token_sequence)
    assert 'token_embedding' in include and 'unembedding.1' in include, "Must include at least token embedding and unembedding"

    def _zero_ablated(xs, ys):
        toks = to_token_sequence(xs, ys)
        B, T, V = toks.shape

        x = model.token_sequence_transformer.token_embedding(toks)    # B T V @ . V C -> B T C

        if 'postn_embedding' in include:
            x += model.token_sequence_transformer.postn_embedding.weight.T[:T, :]

        if 'blocks.0' in include:
            x += model.token_sequence_transformer.blocks[0](x)
        else: # Mean-ablate
            x += model.token_sequence_transformer.blocks[0](x).mean(dim=0, keepdim=True)
        
        if 'blocks.1' in include:
            x += model.token_sequence_transformer.blocks[1](x)
        else: # Mean-ablate
            x += model.token_sequence_transformer.blocks[1](x).mean(dim=0, keepdim=True)

        if 'unembedding.0' in include:
            x = model.token_sequence_transformer.unembedding[0](x)
        
        ytoks = model.token_sequence_transformer.unembedding[1](x)
        del toks, x

        return from_predicted_token_sequence(ytoks)

    return _zero_ablated
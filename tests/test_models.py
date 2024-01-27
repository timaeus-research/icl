import torch
import torch_testing as tt

from icl.regression.model import from_token_sequence, to_token_sequence

# # # TEST MODELS MODULE (incomplete)


def test_to_token_sequence():
    xs = torch.asarray(
        [ [ [1,2,3], [2,3,4], [6,5,4], ] ]
    )
    ys = torch.asarray(
        [ [ [1],     [2],     [6],     ] ]
    )
    toks = to_token_sequence(xs, ys)
    expected_toks = torch.asarray(
      [ [ [ 0, 1, 2, 3, ]
        , [ 1, 0, 0, 0, ]
        , [ 0, 2, 3, 4, ]
        , [ 2, 0, 0, 0, ]
        , [ 0, 6, 5, 4, ]
        , [ 6, 0, 0, 0, ]
        ] ]
    )
    tt.assert_equal(toks, expected_toks)
    

def test_from_token_sequence():
    toks = torch.asarray(
      [ [ [ 0, 1, 2, 3, ]
        , [ 1, 0, 0, 0, ]
        , [ 0, 2, 3, 4, ]
        , [ 2, 0, 0, 0, ]
        , [ 0, 6, 5, 4, ]
        , [ 6, 0, 0, 0, ]
        ] ]
    )
    xs, ys = from_token_sequence(toks)
    expected_xs = torch.asarray(
        [ [ [1,2,3], [2,3,4], [6,5,4], ] ]
    )
    expected_ys = torch.asarray(
        [ [ [1],     [2],     [6],     ] ]
    )
    tt.assert_equal(xs, expected_xs)
    tt.assert_equal(ys, expected_ys)


def test_to_from_token_sequence_roundtrip():
    xs = torch.randn(10, 16, 8)
    ys = torch.randn(10, 16, 1)
    toks = to_token_sequence(xs, ys)
    xs_, ys_ = from_token_sequence(toks)
    tt.assert_equal(xs, xs_)
    tt.assert_equal(ys, ys_)



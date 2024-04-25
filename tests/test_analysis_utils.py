import re

from icl.analysis.utils import match_template


def test_match_template():
    # Test case 1: Exact match
    template1 = 'token_sequence_transformer.blocks.0.attention.weight'
    string1 = 'token_sequence_transformer.blocks.0.attention.weight'
    assert match_template(template1, string1) == True

    # Test case 2: Single wildcard match
    template2 = 'token_sequence_transformer.blocks.*.attention.weight'
    string2 = 'token_sequence_transformer.blocks.1.attention.weight'
    assert match_template(template2, string2) == True

    # Test case 3: Double wildcard match
    template3 = 'token_sequence_transformer.blocks.**.weight'
    string3 = 'token_sequence_transformer.blocks.2.attention.attention.weight'
    assert match_template(template3, string3) == True

    # Test case 4: No match
    template4 = 'token_sequence_transformer.blocks.*.ffn.weight'
    string4 = 'token_sequence_transformer.blocks.1.attention.weight'
    assert match_template(template4, string4) == False

    # Test case 5: Multiple wildcards
    template5 = 'token_sequence_transformer.*.*.attention.**'
    string5 = 'token_sequence_transformer.blocks.2.attention.bias'
    assert match_template(template5, string5) == True

    # Test case 6: Dot in the string
    template6 = 'token_sequence_transformer.blocks.*.attention.**'
    string6 = 'token_sequence_transformer.blocks.0.attention.attention.weight'
    assert match_template(template6, string6) == True

    # Test case 7: Dot in the template
    template7 = 'token_sequence_transformer.blocks.*.attention.*'
    string7 = 'token_sequence_transformer.blocks.1.attention.weight'
    assert match_template(template7, string7) == True

    # Test case 8: Empty template and string
    template8 = ''
    string8 = ''
    assert match_template(template8, string8) == True

    # Test case 9: Empty template, non-empty string
    template9 = ''
    string9 = 'token_sequence_transformer.blocks.0.attention.weight'
    assert match_template(template9, string9) == False

    # Test case 10: Non-empty template, empty st
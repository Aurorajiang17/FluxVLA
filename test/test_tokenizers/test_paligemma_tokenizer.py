# Copyright 2026 Limx Dynamics
"""Offline behavior checks for legacy and preformatted PaliGemma prompts."""

import io

import numpy as np
import pytest
import sentencepiece as spm

from fluxvla.tokenizers import paligemma_tokenizer as module
from fluxvla.transforms.transform_prompts import ProcessPrompts


@pytest.fixture
def recorded_tokenizer(tmp_path, monkeypatch):
    asset = tmp_path / 'tokenizer.model'
    asset.write_bytes(b'test-processor-placeholder')
    calls = []

    class Recorder:

        def encode(self, text, **kwargs):
            calls.append((text, kwargs))
            return [2, 41, 42] if kwargs.get('add_bos') else [17]

    monkeypatch.setattr(module, 'maybe_download', lambda *a, **k: asset)
    monkeypatch.setattr(module.sentencepiece, 'SentencePieceProcessor',
                        lambda **kwargs: Recorder())
    return calls


@pytest.mark.parametrize('explicit_default', [False, True])
def test_default_keeps_legacy_cleanup_and_separate_newline(
        recorded_tokenizer, explicit_default):
    kwargs = {'preformatted': False} if explicit_default else {}
    tokenizer = module.PaligemmaTokenizer(**kwargs)
    result = tokenizer('  Pick_up\nCup  ', state=np.zeros(8))
    assert recorded_tokenizer == [('Pick up Cup', {
        'add_bos': True
    }), ('\n', {})]
    assert result == dict(input_ids=[2, 41, 42, 17], attention_mask=[True] * 4)


@pytest.mark.parametrize(
    'prompt',
    ['Task: Pick cup, State: 0 127;\nAction: ', '', '  A_B\tC\r\n ', '拿起杯子 😀'])
def test_preformatted_preserves_the_entire_prompt(recorded_tokenizer, prompt):
    tokenizer = module.PaligemmaTokenizer(preformatted=True)
    result = tokenizer(prompt, add_special_tokens=True)
    assert recorded_tokenizer == [(prompt, {'add_bos': True})]
    assert result == dict(input_ids=[2, 41, 42], attention_mask=[True] * 3)


@pytest.mark.parametrize('value', [None, 0, 1, 'false'])
def test_preformatted_requires_a_boolean(recorded_tokenizer, value):
    with pytest.raises(TypeError, match='preformatted'):
        module.PaligemmaTokenizer(preformatted=value)


@pytest.mark.parametrize('max_len, tokens, mask', [
    (5, [2, 41, 42, 0, 0], [True, True, True, False, False]),
    (2, [2, 41], [True, True]),
])
def test_process_prompts_owns_token_padding_and_truncation(
        recorded_tokenizer, max_len, tokens, mask):
    transform = ProcessPrompts(
        tokenizer=dict(type='PaligemmaTokenizer', preformatted=True),
        max_len=max_len)
    output = transform({'prompt': 'Already formatted;\nAction: '})
    np.testing.assert_array_equal(output['lang_tokens'], tokens)
    np.testing.assert_array_equal(output['lang_masks'], mask)
    assert recorded_tokenizer == [('Already formatted;\nAction: ', {
        'add_bos': True
    })]


@pytest.mark.parametrize('preformatted', [False, True])
def test_local_sentencepiece_asset_survives_save_and_explicit_mode_reload(
        tmp_path, preformatted):
    # A tiny, deterministic local asset avoids model downloads in CPU CI.
    writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter([
            'Task Pick cup State 0 127 255 Action', 'Pick_up cup',
            'Test prompt'
        ]),
        model_writer=writer,
        model_type='char',
        vocab_size=64,
        hard_vocab_limit=False,
        num_threads=1,
        minloglevel=2)
    asset = tmp_path / 'source.model'
    asset.write_bytes(writer.getvalue())
    tokenizer = module.PaligemmaTokenizer(
        model_path=str(asset), preformatted=preformatted)
    prompt = 'Task: Pick cup, State: 0 127;\nAction: '
    before = tokenizer(prompt)
    saved = tmp_path / 'saved'
    tokenizer.save_pretrained(str(saved))
    assert (saved / 'tokenizer.model').read_bytes() == asset.read_bytes()
    reloaded = module.PaligemmaTokenizer(
        model_path=str(saved), preformatted=preformatted)
    assert reloaded(prompt) == before

# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import numpy as np
import sentencepiece

from fluxvla.engines import TOKENIZERS
from fluxvla.engines.utils.download_utils import maybe_download


@TOKENIZERS.register_module()
class PaligemmaTokenizer:
    """Encode text prompts with the PaliGemma SentencePiece model.

    State formatting belongs to the prompt transform. This tokenizer does
    not encode or decode actions and does not use FAST.

    Args:
        download_path (str): Fallback path or URL for the tokenizer model.
            Defaults to "gs://big_vision/paligemma_tokenizer.model".
        model_path (str | None): Local model file or saved tokenizer directory.
        preformatted (bool): Encode the complete prompt with BOS, without
            cleanup or an appended newline. Defaults to False for legacy
            prompt handling. Select this mode explicitly when reloading.
    """

    def __init__(
            self,
            download_path: str = 'gs://big_vision/paligemma_tokenizer.model',
            model_path: str = None,
            *args,
            preformatted: bool = False,
            **kwargs):

        if not isinstance(preformatted, bool):
            raise TypeError('preformatted must be a bool')
        self.preformatted = preformatted

        # Evaluation runners pass the checkpoint-local tokenizer directory via
        # ``model_path``.  Prefer that artifact over the remote fallback so a
        # self-contained training work directory can be evaluated offline.
        if model_path is not None:
            path = Path(model_path).expanduser()
            if path.is_dir():
                path = path / 'tokenizer.model'
            download_path = path.as_posix()

        path = maybe_download(download_path, gs={'token': 'anon'})
        with path.open('rb') as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read())

    def tokenize(self,
                 prompt: str,
                 state: np.ndarray | None = None,
                 **kwargs) -> dict[str, list]:
        """Tokenize a prompt and return unpadded IDs and attention masks.

        Args:
            prompt (str): The task prompt to be tokenized.
            state (np.ndarray | None): Unused compatibility argument.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: ``input_ids`` and ``attention_mask`` lists.
        """
        if self.preformatted:
            tokens = self._tokenizer.encode(prompt, add_bos=True)
        else:
            cleaned_text = prompt.strip().replace('_', ' ').replace('\n', ' ')
            tokens = self._tokenizer.encode(
                cleaned_text, add_bos=True) + self._tokenizer.encode('\n')

        token_mask = [True] * len(tokens)
        return dict(input_ids=tokens, attention_mask=token_mask)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def save_pretrained(self, save_directory: str):
        """Save the tokenizer to a directory.

        Args:
            save_directory (str): The directory to save the tokenizer.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. save sentencepiece model file
        spm_file = save_dir / 'tokenizer.model'
        # use serialized_model_proto is the most reliable way,
        # not dependent on "loading from file or bytes"
        with spm_file.open('wb') as f:
            f.write(self._tokenizer.serialized_model_proto())

        # 2. optional: save some meta information (for later loading)
        config = {
            'tokenizer_class': 'PaligemmaTokenizer',
            'spm_file': 'tokenizer.model'
        }
        with (save_dir / 'tokenizer_config.json').open(
                'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

"""LMM-PCQA's score of a point cloud in memory, as the release's ``pcqa_eval.py`` scores the
folder of a point cloud's projections, with the released Q-Align weights::

    Q = Σ_k softmax(ℓ)_k w_k,   w = (1, 0.75, 0.5, 0.25, 0),

``ℓ`` the logits of the words excellent, good, fair, poor and bad after the prompt, read for
the six projections of :mod:`lmm_pcqa_projections` as one image sequence; ``wa5`` is the
release's. Runs in the fork's venv (``setup_venv.sh``), where Q-Align is installed.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from q_align.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from q_align.conversation import conv_templates
from q_align.mm_utils import expand2square, get_model_name_from_path, tokenizer_image_token
from q_align.model.builder import load_pretrained_model
from q_align.model.tokenization_llama import LlamaTokenizer
from torch import Tensor

from lmm_pcqa_projections import projections
from pcqa_eval import wa5

# pcqa_eval.py's conversation template, question and answer stem.
_TEMPLATE = "mplug_owl2"
_QUESTION = "How would you rate the quality of this image?"
_ANSWER = " The quality of the image is"
# The words wa5 weights, as pcqa_eval.py names them.
_LEVELS = ("excellent", "good", "fair", "poor", "bad")
# The byte range the background colour is scaled to, as pcqa_eval.py scales it.
_BYTE = 255


class CloudScorer:
    """The released network and the authors' projections, scoring one cloud at a time."""

    def __init__(self, device: torch.device, weights: Path) -> None:
        """Load the released network, the prompt and the level words' ids.

        Args:
            device: The CUDA device the network runs on; one without an index means the
                current one.
            weights: Directory of the released weights (``q-future/q-align-pcqa-*``).
        """
        index = torch.cuda.current_device() if device.index is None else device.index
        self._device = torch.device(device.type, index)
        name = str(weights)
        tokenizer, self._model, self._processor, _ = load_pretrained_model(
            name,
            model_base=None,
            model_name=get_model_name_from_path(name),
            device=str(self._device),
        )
        self._input_ids = _prompt_ids(tokenizer).to(self._device)
        self._level_ids = [word[1] for word in tokenizer(list(_LEVELS))["input_ids"]]
        self._background = tuple(int(x * _BYTE) for x in self._processor.image_mean)

    def __call__(self, points: np.ndarray, colours: np.ndarray) -> float:
        """Q of a cloud, in 0..1, higher meaning better.

        Args:
            points: Coordinates, ``N × 3``.
            colours: Colours in 0..1, ``N × 3``.
        """
        views = projections(points, colours)
        return self.score_views(views)

    def score_views(self, views: list[np.ndarray]) -> float:
        """Q of a cloud's projections, as pcqa_eval.py scores the image files of one cloud.

        Args:
            views: The projections, RGB ``uint8``, in the authors' order.
        """
        images = [expand2square(Image.fromarray(view), self._background) for view in views]
        pixels = self._processor.preprocess(images, return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            output = self._model(self._input_ids, images=[pixels.half().to(self._device)])
            last = output["logits"][:, -1].mean(0)
            logits = {level: last[index].item() for level, index in zip(_LEVELS, self._level_ids)}
        score = float(wa5(logits))
        return score


def _prompt_ids(tokenizer: LlamaTokenizer) -> Tensor:
    """pcqa_eval.py's prompt as token ids, the image's place marked, ``[1, L]``.

    Args:
        tokenizer: The released weights' tokenizer.
    """
    conversation = conv_templates[_TEMPLATE].copy()
    conversation.append_message(conversation.roles[0], f"{_QUESTION}\n{DEFAULT_IMAGE_TOKEN}")
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt() + _ANSWER
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    batch = ids.unsqueeze(0)
    return batch

"""Score point-cloud projections with a Q-Align PCQA checkpoint (LMM-PCQA projection branch).

Mirrors the scoring part of pcqa_eval.py (prompt, five-token logits, wa5) but takes ordinary
image folders instead of the authors' hard-coded dataset paths. Run with venvs/qalign/bin/python
(third_party/scripts/setup_qalign_venv.sh) and HF_HOME pointing at the checkpoint cache.

  one object:   python score_views.py --views dir_with_pngs
  MM-PCQA-style: python score_views.py --csv subset.csv --proj_dir wpc_projections   (name,mos rows;
                 each object is a folder <name>/0.png ... 3.png) -> per-object scores + SROCC/PLCC
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from q_align.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from q_align.conversation import conv_templates
from q_align.mm_utils import get_model_name_from_path, tokenizer_image_token
from q_align.model.builder import load_pretrained_model

TOKS = ["excellent", "good", "fair", "poor", "bad"]
WEIGHTS = np.array([1, 0.75, 0.5, 0.25, 0.0])


def wa5(logits):
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return float(np.inner(probs, WEIGHTS))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    size = max(width, height)
    result = Image.new(pil_img.mode, (size, size), background_color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="q-future/q-align-pcqa-wpc")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--views", help="folder of projection images of one object")
    ap.add_argument("--csv", help="csv with name,mos columns (MM-PCQA style)")
    ap.add_argument("--proj_dir", help="folder with <name>/<i>.png projections for the csv rows")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    torch.nn.Linear.reset_parameters = lambda self: None  # as in pcqa_eval.py: skip random init
    torch.nn.LayerNorm.reset_parameters = lambda self: None
    model_name = get_model_name_from_path(a.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(a.model_path, None, model_name, False, False, device=a.device)
    conv = conv_templates["mplug_owl2"].copy()
    conv.append_message(conv.roles[0], "How would you rate the quality of this image?\n" + DEFAULT_IMAGE_TOKEN)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " The quality of the image is"
    ids = [tokenizer(t)["input_ids"][1] for t in TOKS]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(a.device)
    bg = tuple(int(x * 255) for x in image_processor.image_mean)

    def score(files):
        imgs = [expand2square(Image.open(f).convert("RGB"), bg) for f in files]
        px = image_processor.preprocess(imgs, return_tensors="pt")["pixel_values"].half().to(a.device)
        with torch.inference_mode():
            out = model(input_ids, images=[px])["logits"][:, -1].mean(0)
        logits = np.array([out[i].item() for i in ids])
        return wa5(logits), dict(zip(TOKS, np.round(logits, 3)))

    if a.views:
        files = sorted(glob.glob(os.path.join(a.views, "*.png")) + glob.glob(os.path.join(a.views, "*.jpg")))
        s, lg = score(files)
        print(f"{a.views}: {len(files)} views -> score {s:.4f} (0..1, higher is better)  logits {lg}")
    if a.csv:
        import csv
        rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
        rows = rows[: a.limit] if a.limit else rows
        prs, gts = [], []
        for r in rows:
            files = sorted(glob.glob(os.path.join(a.proj_dir, r["name"], "*.png")))
            s, _ = score(files)
            prs.append(s); gts.append(float(r["mos"]))
            print(f"{r['name']:32s} views={len(files)} score={s:.4f} mos={gts[-1]:.2f}")
        print(f"n={len(prs)} SROCC={spearmanr(prs, gts)[0]:.4f} PLCC={pearsonr(prs, gts)[0]:.4f}")


if __name__ == "__main__":
    main()

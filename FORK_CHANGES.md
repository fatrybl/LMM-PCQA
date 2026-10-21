# How this fork differs from the original LMM-PCQA code

**Original:** [Q-Future/LMM-PCQA](https://github.com/Q-Future/LMM-PCQA), commit `97e99eb`, the
code of *LMM-PCQA: Assisting Point Cloud Quality Assessment with LMM* (Zhang et al., ACM MM 2024).
**Fork:** [fatrybl/LMM-PCQA](https://github.com/fatrybl/LMM-PCQA), branch `vape-baselines`.
`git diff 97e99eb` shows every change.

The fork serves as a baseline in the comparison of the VAPE project, where the released network
scores a LiDAR point cloud. The method is unchanged: the projections, the prompt, the network, its
weights and the score are the authors'. The fork adds a scorer of a cloud in memory and the venv it
runs in.

## 1. What the release holds

The paper scores a cloud as `Q = SVR(F_L ⊕ F_S)`: the probabilities of five quality words an LMM
gives six projections (`F_L`), and statistics of linearity and planarity (`F_S`), regressed by an
SVR trained per fold. The release (`pcqa_eval.py`) is the LMM branch only, Q-Align's scorer `Q = Σ
softmax(ℓ)_k w_k`, `w = (1, 0.75, 0.5, 0.25, 0)` for excellent … bad; the structural features, the
SVR and the projection code were not released. Its prompt is Q-Align's image prompt, not the paper's
point-cloud prompt, and it is the one the released weights were trained with (the conversations of
the authors' `splits/train_WPC.json`).

## 2. Changes

- **`lmm_pcqa_projections.py` (new): the authors' projections in memory.** The authors distribute
  their projections as a dataset
  ([zhangzicheng/pcqa_dataset_projections](https://huggingface.co/datasets/zhangzicheng/pcqa_dataset_projections))
  and give the recipe in GMS-3DQA's README (same first author, 2024-03-05): MM-PCQA's
  `utils/get_projections.py` (Open3D's legacy visualiser, a hidden 1080 × 1920 window, the camera
  turned before each capture, the white background cropped) with six turns, `(0, 0)`, three of
  `(90°, 0)`, `(0, 90°)` and `(0, 180°)`. The module runs that code on a cloud in memory and returns
  the views `cv2.imwrite` would store and PIL read back (PNG is lossless). It renders with the
  Open3D 0.20 of VAPE's environment.
- **`score_cloud.py` (new): `CloudScorer(device, weights)(points, colours)`**, `pcqa_eval.py`'s
  scoring of one cloud: its conversation, question and answer stem, the level words' ids, its
  padding to a square with the CLIP mean colour, the CLIP preprocessing in fp16, all six views as
  one image sequence, the mean of the last-position logits over the batch of one, and its own `wa5`.
  It does not call `disable_torch_init`, which patched `torch.nn` for the whole process to speed up
  building the model; transformers 5 builds on the meta device and loads every weight, so the result
  is the same.
- **`venv_requirements.txt`, `setup_venv.sh` (new): the venv the scorer runs in.** VAPE's
  environment through `--system-site-packages` (torch, Open3D, OpenCV, VAPE), the pinned
  transformers 5.20.2 and its tokenizers, sentencepiece, and the Q-Align fork (ported to
  transformers 5, see its FORK_CHANGES.md) installed beside this one; nothing of VAPE's
  environment is replaced.
- **`test_projections.py` (new):** the in-memory views against the recipe's PNG files.
- `score_views.py`, an earlier command-line scorer of image folders, removed; `weights/` ignored.

## 3. Weights

`weights/q-align-pcqa-wpc/` holds [q-future/q-align-pcqa-wpc](https://huggingface.co/q-future/q-align-pcqa-wpc)
at revision `4d7bfe46d31e58ace69370909212e8bd7d89190b`, the four shards with the sha256 of the Hub's
LFS records (`9303daec…`, `b7a2c5fa…`, `dc5b97de…`, `934cc9d4…`; tokenizer.model `9e556afd…`). Its
`trainer_state.json` and `training_args.bin` record the training: 593 clouds, the authors'
`train_WPC.json` (test objects banana, cauliflower, mushroom and pineapple; one banana cloud is in
the training split), 6 epochs where the paper states 2, batch 64 on 8 GPUs, learning rate 2·10⁻⁵,
from `q-future/one-align`. The weights are therefore one fold's model, not the paper's five-fold
results. The SJTU and WPC2.0 checkpoints were trained on 336 and 318 clouds; the WPC2.0 split shares
the object `ship` between training and test.

## 4. Verification

- **Projections.** On WPC's `bag_level_7`, the recipe on Open3D 0.20 gives the crop sizes of the
  authors' published views in 6 of 6; 95.2–95.9 % of their pixels are equal bit for bit, and
  99.2–99.8 % of the others within 8/255, depending on the view. The score of the re-render is
  5.402627·10⁻⁴ against 5.402706·10⁻⁴ for the published views. In memory and through files the views
  are equal bit for bit on 180 views of 30 LiDAR clouds (ten ONCE samples of VAPE's comparison) and
  in the test; two runs give the same bytes. A window kept from one cloud to the next is not
  equivalent (354 of 360 views differed), so every cloud gets a new window, as in the release.
- **Scores.** The reference is `pcqa_eval.py`'s scoring with Q-Align as released under transformers
  4.36.1 (eager attention, as under the release's torch 2.0.1). On 32 sets of projections (the
  authors' and the re-rendered `bag_level_7`, the 30 LiDAR clouds) the scorer gives the reference's
  score bit for bit, from the image files and from the clouds. A second, freshly installed
  transformers 4.36.1 reproduced that reference bit for bit. In VAPE's comparison, the values a run
  recorded at the start and the truth of 29 sample-levels of its paper ladder equal those of the
  release's own pipeline (the recipe's image files scored by `pcqa_eval.py`'s statements) in 58
  of 58.
- **Cost on an RTX 3090:** 0.64 s per cloud on the 30 clouds (projections 0.27 s, preprocessing
  0.07 s, network 0.29 s), 0.60 to 0.66 s in the searches of VAPE's comparison; 15.8 GiB of GPU
  memory.

## 5. Kept as released

`pcqa_eval.py`, the network, the weights and the prompt.

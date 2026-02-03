# VISTA-Bench

We introduce **VISTA-Bench**, a systematic benchmark spanning multimodal perception, reasoning, and unimodal understanding. It evaluates *visualized text understanding* by contrasting **pure-text** and **visualized-text (VT)** questions under controlled rendering conditions.

## Dataset at a glance

- **Size:** 1,500 instances
- **Composition:** predominantly multiple-choice questions (MCQ), with a small portion of open-ended queries
- **Task taxonomy (4 primary categories):**
  - **Unimodal Knowledge:** 500
  - **Multimodal Knowledge:** 400
  - **Multimodal Perception:** 300
  - **Multimodal Reasoning:** 300

This hierarchy is designed to diagnose whether failures come from **visual text perception** (reading pixels) versus downstream **reasoning/knowledge**, especially when no symbolic text is available.

## Repository structure

```text
VISTA-Bench/
├─ images/            # original images
├─ questions/         # rendered question/option images for the VT setting
├─ VLMEvalKit/        # evaluation toolkit
├─ VISTA-Bench.tsv    # dataset index (currently identical to the VT variant)
└─ VISTA-Bench-VT.tsv # dataset index (currently identical; different filename for compatibility)
```

> **Note:** `VISTA-Bench.tsv` and `VISTA-Bench-VT.tsv` are currently identical; we keep both filenames for compatibility and will refine the organization later.

## Data

- `images/`: original images used by multimodal instances
- `questions/`: rendered question/option images for the visualized-text (VT) setting
- `*.tsv`: dataset metadata and file paths used for evaluation

## Evaluation

We evaluate VISTA-Bench with `VLMEvalKit/`. Before running evaluation, we recommend converting the TSV file(s) into a normalized format with absolute image paths.

### 1) Convert TSV to normalized paths

We provide a helper script:

- `VISTA-Bench/VLMEvalKit/utils/convert_data_file.py`

This script:
- converts the TSV encoding to UTF-8
- normalizes path separators (`\` → `/`)
- renames `options_A/B/C/D` to `A/B/C/D` when needed
- converts `image_path` and `question_image_path` into a bracket-style multi-path string

Example:

```bash
python VISTA-Bench/VLMEvalKit/utils/convert_data_file.py \
  --in   VISTA-Bench/VISTA-Bench.tsv \
  --out  VISTA-Bench/VISTA-Bench_norm.tsv \
  --image-prefix /ABS/PATH/TO/VISTA-Bench \
  --strip-token ""
```

### 2) Running evaluation

Run `run.py` with the dataset name for each modality.

Pure-text:
```bash
python run.py \
  --data VISTA-Bench \
  --model llava_v1.5_7b \
  --verbose
```

Visualized-text (VT):
```bash
python run.py \
  --data VISTA-Bench-VT \
  --model llava_v1.5_7b \
  --verbose
```

## Citation

To be added.

## License

To be added.

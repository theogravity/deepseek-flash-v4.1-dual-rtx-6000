# Third-party notices

This repository's original files are MIT-licensed (see `LICENSE`). It also contains modified copies of third-party code, which stay under
their own licenses:

- `patches/engram-nvme/patched/engram.py`, `patches/engram-nvme/patched/ep_weight_filter.py`, `patches/api-auth/patched/authenticate.py` — modified copies
  of files from [vLLM](https://github.com/vllm-project/vllm) (Apache License 2.0, Copyright the vLLM contributors; text in `LICENSES/Apache-2.0.txt`).
  The `.diff` files next to them show exactly what was changed. The changes are small additions marked with comments; the rest is upstream code.
- `bench/quality_probe_sandboxed.py` — a modified copy of `bench/quality-probe.py` from Diffbot's recipe (MIT, Copyright (c) 2026 Diffbot; text in
  `LICENSES/Diffbot-recipe-MIT.txt`). The change replaces host execution of model-generated code with a sandboxed container run.
- `bench/run_quality.sh` calls Diffbot's recipe scripts (`bench/len_sweep.py`, `test-toolcall.py`, `test-image.py`, `needle-test.py`, `fetch-evaldata.sh`)
  from your own checkout of the recipe; they are not included here. Their GSM8K and HumanEval datasets are MIT-licensed OpenAI releases.
- `compose/docker-compose.yml` reproduces the environment variables of Diffbot's `serve/serve.sh` (MIT, same license text as above).

Not included, but used inside the images the recipe builds: ExLlamaV3 (MIT) and the vllm-exl3 plugin (AGPL-3.0, see the recipe's notices). The model
weights are Diffbot's pack of DeepSeek V4.1 Flash; see its Hugging Face page for the model license. LMCache (Apache-2.0) is installed from PyPI by
`patches/lmcache/build_image.sh`.

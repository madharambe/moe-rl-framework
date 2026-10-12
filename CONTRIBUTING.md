# Contributing

Thanks for looking. This repository is a research framework, so the most useful
contributions are usually one of:

- **a reproducible result** — a training or evaluation run with its configuration,
  the exact command, and the raw output rather than a summary;
- **a bug fix with a measurement** — what you observed, what you expected, and the
  smallest command that shows the difference;
- **an evaluation suite** for a capability this repo does not cover yet.

## Ground rules

1. **No datasets, no checkpoints, no large binaries in git.** This repository ships
   code. Corpora and weights belong on a data volume or on the Hugging Face Hub.
2. **No secrets, ever** — not in code, not in a config, not in a test fixture. If one
   is ever committed, rotate it: removing it from history is not enough.
3. **State what you measured, and how.** A claim without its command and its raw
   output is a story, not a result. Negative results are welcome and are explicitly
   part of this project's record.
4. **Keep paths parameterised.** Use `RSI_HOME` (and the other environment variables
   the code already reads) rather than absolute paths to your own machine.

## Getting started

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

Install the PyTorch wheel that matches your CUDA driver first — see the note at the
top of `requirements.txt`.

## Before opening a pull request

```bash
python3 -m compileall -q src            # every module must at least import-clean
python3 -c "import ast,glob;[ast.parse(open(f).read()) for f in glob.glob('src/**/*.py',recursive=True)]"
```

If your change touches training or evaluation behaviour, describe in the pull request
what you ran and paste the raw numbers.
#!/usr/bin/env python3
"""Upload a trained checkpoint + model card to the Hugging Face Hub.

Usage
  python tools/upload_to_hf.py --repo <user>/<name> --weights runs/moe.pt --dry-run
  python tools/upload_to_hf.py --repo <user>/<name> --weights runs/moe.pt

Auth: run `hf auth login` first, or export HF_TOKEN. The script never prints the
token. DRY RUN IS THE DEFAULT - nothing is uploaded unless you pass --apply, because
an accidental publish of a half-trained checkpoint is worse than no publish.
"""
import argparse
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, help='e.g. yourname/moe-rl-58m')
    ap.add_argument('--weights', required=True, help='path to the .pt checkpoint')
    ap.add_argument('--card', default='MODEL_CARD.md')
    ap.add_argument('--license', default='LICENSE')
    ap.add_argument('--apply', action='store_true', help='actually upload')
    a = ap.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print('huggingface_hub is not installed: pip install huggingface_hub')
        return 2

    token = os.environ.get('HF_TOKEN')
    api = HfApi(token=token)
    try:
        who = api.whoami()
        print('authenticated as : %s' % who.get('name'))
    except Exception as exc:
        print('NOT AUTHENTICATED (%s).' % type(exc).__name__)
        print('run `hf auth login` or export HF_TOKEN, then retry.')
        return 3

    for p in (a.weights, a.card, a.license):
        if not os.path.exists(p):
            print('missing file: %s' % p)
            return 4
    size_gb = os.path.getsize(a.weights) / 2**30
    print('repo            : %s' % a.repo)
    print('weights         : %s (%.2f GB)' % (a.weights, size_gb))
    print('card / license  : %s / %s' % (a.card, a.license))

    # a guard worth having: a checkpoint this large is almost never intentional
    if size_gb > 20:
        print('REFUSING: %.1f GB is unusually large for this project - check the path.'
              % size_gb)
        return 5

    if not a.apply:
        print()
        print('DRY RUN - nothing uploaded. Re-run with --apply to publish.')
        return 0

    create_repo(a.repo, repo_type='model', exist_ok=True, token=token)
    for local, remote in ((a.card, 'README.md'), (a.license, 'LICENSE'),
                          (a.weights, os.path.basename(a.weights))):
        print('uploading %s -> %s' % (local, remote))
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=a.repo, token=token)
    print()
    print('done: https://huggingface.co/%s' % a.repo)
    return 0


if __name__ == '__main__':
    sys.exit(main())
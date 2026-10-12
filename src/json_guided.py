#!/usr/bin/env python3
# 205u: GUIDED JSON DECODING — a character-level state machine for the
# tool-call JSON emission. At each decode step the candidates are masked
# to tokens whose FIRST CHARACTER keeps the JSON structurally valid
# (fp8.co small-model guide: schema-constrained decoding). Fallbacks:
# empty mask -> unrestricted sampling; the constraint never hard-fails.

_ALPHA = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DIGIT = set("0123456789")
_VALUE_START = _ALPHA | _DIGIT | set('"-{[.')
_WS = set(" \t\n\r")


def _with_ws(s):
    return (s | _WS) if s is not None else None


class GuidedJSON:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stack = []           # top = current container ("obj"/"arr")
        self.depth = 0            # raw { [ depth
        self.in_string = False
        self.esc = False
        self.after_value = False  # next non-space char must be , } ] or :
        self.expect_key = True    # inside obj: expecting " or }
        self.need_colon = False   # a key just closed; expecting :
        self._opened_key = False  # the string that just opened was a key
        self.active = False       # inside a tool-call JSON

    def feed(self, text: str):
        for ch in text:
            if self.in_string:
                if self.esc:
                    self.esc = False
                elif ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.in_string = False
                    if self._opened_key:
                        self.need_colon = True
                        self.expect_key = False
                    else:
                        self.after_value = True
                continue
            if ch == '"':
                self.in_string = True
                self.after_value = False
                self._opened_key = self.expect_key
                self.need_colon = False
                continue
            if ch.isspace():
                continue
            if ch == "{":
                self.depth += 1
                self.stack.append("obj")
                self.after_value = False
                self.expect_key = True
                if self.depth >= 1:
                    self.active = True
                continue
            if ch == "[":
                self.depth += 1
                self.stack.append("arr")
                self.after_value = False
                if self.depth >= 1:
                    self.active = True
                continue
            if ch in "}":
                self.depth = max(0, self.depth - 1)
                if self.stack:
                    self.stack.pop()
                if self.depth == 0:
                    self.active = False
                self.after_value = True
                continue
            if ch in "]":
                self.depth = max(0, self.depth - 1)
                if self.stack:
                    self.stack.pop()
                if self.depth == 0:
                    self.active = False
                self.after_value = True
                continue
            if ch in ",":
                self.after_value = False
                self.expect_key = True
                continue
            if ch == ":":
                self.after_value = False
                self.expect_key = False
                self.need_colon = False
                continue
            self.after_value = True

    def allowed_chars(self):
        if not self.active:
            return None
        if self.in_string:
            return None
        if self.after_value:
            return _with_ws(set(",}]") | (set(":") if not self.stack else set()))
        if self.need_colon:
            return _with_ws({":"})
        if self.expect_key:
            return _with_ws(set('"') | (set("}") if len(self.stack) == 1 and self.depth == 1 else set()))
        return _with_ws(_VALUE_START)

    def engaged(self) -> bool:
        return self.active and not self.in_string


def mask_topk(logits, tok, allowed, top_k=200):
    """mask logits so only tokens whose first char is in allowed survive;
    returns (masked_logits, advise_rand) — advise_rand True when nothing
    survived and the caller should sample unrestricted."""
    import torch
    keep = []
    vals, idx = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
    for v, i in zip(vals.tolist(), idx.tolist()):
        t = tok.decode([i]) or ""
        if t and t[0] in allowed:
            keep.append(i)
    if not keep:
        return logits, True
    m = torch.full_like(logits, float("-inf"))
    m[keep] = logits[keep]
    return m, False


class GuidedSampler:
    """Per-row GuidedJSON machines + logits masking, driven by MoEModel.
    Contract inside model.generate (guided_hook):
      logits[row] = hook.mask(row, logits[row], tokens_so_far)
      hook.feed(row, sampled_token_id)   # after sampling
    mask() decodes the tokens from `tokens_so_far` through its internal
    feed-position so it stays consistent even if the caller truncates."""
    def __init__(self, detok):
        self.detok = detok
        self.machines = {}
        self.fed = {}
        self.fallbacks = 0

    def _m(self, row):
        if row not in self.machines:
            self.machines[row] = GuidedJSON()
            self.fed[row] = 0
        return self.machines[row]

    def mask(self, row, logits, tokens_so_far):
        m = self._m(row)
        # feed any tokens not fed yet
        if len(tokens_so_far) > self.fed[row]:
            tail = tokens_so_far[self.fed[row]:]
            try:
                txt = self.detok(tail) if len(tail) else ""
            except Exception:
                txt = ""
            m.feed(txt)
            self.fed[row] = len(tokens_so_far)
        al = m.allowed_chars()
        if al is None:
            return logits
        mm, advise = mask_topk(logits, self.detok, al)
        if advise:
            self.fallbacks += 1
            return logits
        return mm

    def feed(self, row, token_id):
        # the caller appends after sampling; decode the single token
        m = self._m(row)
        try:
            txt = self.detok([token_id])
        except Exception:
            txt = ""
        m.feed(txt)
        self.fed[row] = self.fed.get(row, 0) + 1

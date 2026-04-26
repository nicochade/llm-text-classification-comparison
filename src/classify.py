"""SLM prompting utilities for zero-shot / few-shot sentiment classification.

The prompts here are lifted almost verbatim from the original coursework
notebook. ``fs_min`` (hardcoded Spanish examples on an English dataset)
is replaced by ``fs_train``, a template that is filled in at runtime
with examples drawn deterministically from the train split.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Prompt templates: (system_prompt, user_template)
# ---------------------------------------------------------------------------
PROMPTS = {
    "zs_strict_en": (
        "You are a sentiment classifier. Reply with EXACTLY one word: "
        "positive or negative. No punctuation, no explanations.",
        'Review:\n"""{text}"""\nLabel:',
    ),
    "zs_strict_es": (
        "Eres un clasificador de sentimiento. Responde con UNA sola palabra "
        "EXACTA: positive o negative. Sin puntuacion ni explicacion.",
        'Resena:\n"""{text}"""\nEtiqueta:',
    ),
    "zs_calibrated": (
        "You are a strict classifier. Prior over classes is equal. If uncertain, "
        "still choose exactly one from: positive, negative. Any other output is invalid.",
        'Text:\n"""{text}"""\nReturn only one of [positive, negative]:',
    ),
    # fs_train is produced at runtime by build_fs_prompt() using examples
    # sampled from the train split.
}


def build_fs_prompt(examples: pd.DataFrame) -> Tuple[str, str]:
    """Build a few-shot (system, user_template) pair from train-split examples.

    ``examples`` must have columns ``review`` and ``sentiment``. Each row
    contributes one demonstration. The user template ends with the new
    review to classify.
    """
    system = (
        "You are a sentiment classifier. Reply with EXACTLY one word: "
        "positive or negative. No punctuation, no explanations."
    )

    demo_lines = []
    for _, row in examples.iterrows():
        demo_lines.append(f'Review: "{row["review"]}" -> {row["sentiment"]}')
    demos_block = "\n".join(demo_lines)

    user = (
        "Examples:\n"
        f"{demos_block}\n"
        "Now classify this review:\n"
        'Review:\n"""{text}"""\nLabel:'
    )
    return system, user


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load the SLM tokenizer and model onto the best available device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
    )
    return tok, model


# ---------------------------------------------------------------------------
# Chat + parsing
# ---------------------------------------------------------------------------
def chat(
    tok,
    model,
    system: str,
    user: str,
    max_new_tokens: int = 1,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> str:
    """Single chat turn. Returns only the newly generated text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok([prompt_text], return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


_LABEL_SET = {"positive", "negative"}
_TAG_RE = re.compile(r"<.*?>")
_LABEL_RE = re.compile(r"\b(positive|negative)\b", flags=re.IGNORECASE)


def parse_label(text: str) -> str:
    """Extract 'positive' / 'negative' from a raw model output, or 'unk'."""
    if not isinstance(text, str):
        return "unk"
    cleaned = _TAG_RE.sub("", text).strip().lower()
    m = _LABEL_RE.search(cleaned)
    if m is None:
        return "unk"
    lab = m.group(1).lower()
    return lab if lab in _LABEL_SET else "unk"


# ---------------------------------------------------------------------------
# Batch classification with explicit input-length cap
# ---------------------------------------------------------------------------
def truncate_to_tokens(tok, text: str, max_input_tokens: int) -> str:
    """Tokenize, truncate, and decode back to a string of at most
    ``max_input_tokens`` tokens. Returns the original text if no
    truncation is requested."""
    if max_input_tokens is None:
        return text
    enc = tok(text, truncation=True, max_length=max_input_tokens, return_tensors=None)
    return tok.decode(enc["input_ids"], skip_special_tokens=True)


def count_truncation_rate(tok, texts: Iterable[str], max_input_tokens: int) -> float:
    """Fraction of texts whose token length exceeds ``max_input_tokens``."""
    truncated = 0
    total = 0
    for t in texts:
        n = len(tok(t, truncation=False, return_tensors=None)["input_ids"])
        if n > max_input_tokens:
            truncated += 1
        total += 1
    return truncated / total if total else 0.0


def classify_batch(
    texts: Iterable[str],
    tok,
    model,
    system_prompt: str,
    user_template: str,
    max_new_tokens: int = 1,
    max_input_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.0,
    desc: str = "classify",
) -> List[str]:
    """Classify a batch of reviews. Returns a list of predicted labels
    ('positive' / 'negative' / 'unk')."""
    preds: List[str] = []
    for t in tqdm(list(texts), desc=desc):
        t_capped = truncate_to_tokens(tok, t, max_input_tokens)
        raw = chat(
            tok,
            model,
            system_prompt,
            user_template.format(text=t_capped),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        preds.append(parse_label(raw))
    return preds

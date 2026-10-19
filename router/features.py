"""Stage 2 — cheap request features (all µs–ms, no model calls).

These feed the hard rules, the heuristic tagger fallback, and the cost prior.
Token counts are rough estimates (chars/4); the ledger uses Bedrock's real counts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Keyword cues per task family (lowercased substring match). Deliberately small
# and readable — the Haiku tagger does the nuanced classification; this is the
# offline fallback and a feature signal.
FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sql":       ("select ", "join ", "group by", "where ", "cohort", "omop", "faers", "query"),
    "deck":      ("slide", "deck", "pptx", "python-pptx", "presentation"),
    "extract":   ("json", "schema", "extract", "fields", "structured"),
    "pv":        ("adverse event", "meddra", "ich e2a", "seriousness", "icsr", "pharmacovigilance"),
    "biostat":   ("sample size", "power", "hazard ratio", "schoenfeld", "alpha", "type i error"),
    "code":      ("def ", "import ", "pytest", "function", "log2fc", "matrix", "algorithm"),
    "design":    ("adaptive design", "bayesian", "boin", "dose-finding", "operating characteristic", "simulation"),
    "litreview": ("literature", "systematic review", "meta-analysis", "pubmed", "citations"),
    "summarize": ("summarize", "summary", "action items", "minutes", "recap", "transcript"),
    "chat":      ("explain", "rewrite", "plain language", "lay summary", "what is", "how do"),
}

_CODE_FENCE = re.compile(r"```")
_CHARS_PER_TOKEN = 4  # rough English estimate


@dataclass
class Features:
    tokens_in: int
    n_turns: int
    n_tools: int
    has_tool_result: bool
    code_fences: int
    n_images: int
    n_documents: int
    max_tokens: int
    thinking: bool
    system_text: str
    user_text: str
    keyword_hits: dict[str, int] = field(default_factory=dict)


def _system_text(payload: dict) -> str:
    system = payload.get("system", "")
    if isinstance(system, list):
        return " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    return system or ""


def _iter_blocks(payload: dict):
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    yield msg.get("role"), b
        elif isinstance(content, str):
            yield msg.get("role"), {"type": "text", "text": content}


def extract_features(payload: dict) -> Features:
    system_text = _system_text(payload)
    texts = [system_text]
    n_images = n_documents = 0
    has_tool_result = False
    last_user_text = ""
    for role, b in _iter_blocks(payload):
        t = b.get("type")
        if t == "text":
            texts.append(b.get("text", ""))
            if role == "user":
                last_user_text = b.get("text", "")
        elif t == "image":
            n_images += 1
        elif t == "document":
            n_documents += 1
        elif t == "tool_result":
            has_tool_result = True
        elif t == "tool_use":
            pass
    blob = "\n".join(texts)
    low = blob.lower()
    keyword_hits = {
        fam: sum(low.count(kw) for kw in kws) for fam, kws in FAMILY_KEYWORDS.items()
    }
    return Features(
        tokens_in=max(1, len(blob) // _CHARS_PER_TOKEN),
        n_turns=len(payload.get("messages", [])),
        n_tools=len(payload.get("tools", []) or []),
        has_tool_result=has_tool_result,
        code_fences=len(_CODE_FENCE.findall(blob)) // 2,
        n_images=n_images,
        n_documents=n_documents,
        max_tokens=int(payload.get("max_tokens", 0) or 0),
        thinking=bool(payload.get("thinking")),
        system_text=system_text,
        user_text=last_user_text,
        keyword_hits=keyword_hits,
    )

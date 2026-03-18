from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD


@dataclass
class AttackMatch:
    red_team_id: str
    category: str
    severity: str
    similarity_score: float
    attack_intent: str


class SemanticSearch:
    """Semantic similarity search over a red-team adversarial prompt dataset.

    Uses sentence-transformers to embed prompts and
    computes cosine similarity at query time.
    """

    def __init__(self) -> None:
        self._model: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL)
        self._prompts: list[dict] = []
        self._embeddings: np.ndarray | None = None

    def load_dataset(self, path: str) -> None:
        """Load *red_team_dataset.json* and pre-compute prompt embeddings.

        Single-turn entries use the ``"prompt"`` field.
        Multi-turn entries (``"turns"`` array) use their final (most adversarial)
        turn as the representative text for embedding.
        """
        with open(path, encoding="utf-8") as fh:
            dataset = json.load(fh)

        self._prompts = dataset.get("prompts", [])
        texts: list[str] = []
        for p in self._prompts:
            if "prompt" in p:
                texts.append(p["prompt"])
            elif "turns" in p and p["turns"]:
                # Use the last turn — typically the pivot to adversarial intent
                texts.append(p["turns"][-1])
            else:
                texts.append("")  # fallback; won't reach threshold in practice
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)

    def find_matches(self, query: str) -> list[AttackMatch]:
        """Return all red-team entries with cosine similarity >= 0.75,
        sorted by similarity descending."""
        if self._embeddings is None or not self._prompts:
            raise RuntimeError(
                "Dataset not loaded. Call load_dataset() before find_matches()."
            )

        query_vec = self._model.encode([query], normalize_embeddings=True)
        # dot product of unit vectors == cosine similarity
        sims: np.ndarray = (query_vec @ self._embeddings.T)[0]

        matches: list[AttackMatch] = []
        for idx, sim in enumerate(sims):
            if float(sim) >= SIMILARITY_THRESHOLD:
                p = self._prompts[idx]
                matches.append(
                    AttackMatch(
                        red_team_id=p["id"],
                        category=p["category"],
                        severity=p["severity"],
                        similarity_score=round(float(sim), 4),
                        attack_intent=p.get("attack_intent", ""),
                    )
                )
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches

    def is_adversarial(self, query: str) -> bool:
        """Return True if any red-team match is found above the threshold."""
        return len(self.find_matches(query)) > 0

"""GRAPH condition: episodic entity graph.

Nodes are vocabulary entities; an edge is created when two entities co-occur
within a short turn window inside one session (an "episode"). Edges carry the
set of session ids, a recency value, and a partner-engagement flag (True when
the other speaker's reply echoed any of the co-occurring entities, i.e. the
content was substantively responded to rather than merely mentioned).

retrieve(targets) ranks candidate binding nodes for a set of board targets:
same-episode coverage of multiple targets scores high; nodes with many
unrelated neighbors take a generic-association penalty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from ..agents.llm import approx_tokens
from ..engine.wordlist import BINDING_WORD_POOL, BOARD_WORD_POOL
from ..history.plants import DyadHistory
from ..history.validate import tokenize
from .base import MemoryPayload

DEFAULT_VOCAB = set(BOARD_WORD_POOL) | set(BINDING_WORD_POOL)

# score weights
W_SAME_EPISODE_PAIR = 3.0
W_COVERAGE = 1.0
W_ENGAGED = 0.5
W_RECENCY = 0.1
W_GENERIC_PENALTY = 0.2


@dataclass
class Candidate:
    word: str
    score: float
    covered_targets: list[str] = field(default_factory=list)
    episodes: list[int] = field(default_factory=list)
    engaged: bool = False


class EpisodicGraph:
    def __init__(self, g: nx.Graph, n_sessions: int):
        self.g = g
        self.n_sessions = n_sessions

    def retrieve(self, targets: list[str], top_n: int = 5) -> list[Candidate]:
        targets = [t.lower() for t in targets if self.g.has_node(t.lower())]
        if not targets:
            return []
        candidates: list[Candidate] = []
        for node in self.g.nodes:
            if node in targets:
                continue
            covered = [t for t in targets if self.g.has_edge(node, t)]
            if not covered:
                continue
            same_episode_pairs = 0
            for i, t1 in enumerate(covered):
                for t2 in covered[i + 1:]:
                    if (self.g.edges[node, t1]["sessions"]
                            & self.g.edges[node, t2]["sessions"]):
                        same_episode_pairs += 1
            engaged_edges = sum(
                1 for t in covered if self.g.edges[node, t]["partner_engaged"])
            recency = max(self.g.edges[node, t]["recency"] for t in covered)
            recency_norm = recency / max(1, self.n_sessions - 1)
            generic_penalty = W_GENERIC_PENALTY * max(
                0, self.g.degree(node) - len(covered))
            score = (W_SAME_EPISODE_PAIR * same_episode_pairs
                     + W_COVERAGE * len(covered)
                     + W_ENGAGED * engaged_edges
                     + W_RECENCY * recency_norm
                     - generic_penalty)
            episodes = sorted(
                set().union(*(self.g.edges[node, t]["sessions"] for t in covered)))
            candidates.append(Candidate(
                word=node, score=round(score, 4), covered_targets=covered,
                episodes=episodes, engaged=engaged_edges > 0))
        candidates.sort(key=lambda c: (-c.score, c.word))
        return candidates[:top_n]

    def to_payload(self) -> MemoryPayload:
        """Human/LLM-readable rendering of the episodic graph, grouped by session."""
        by_session: dict[int, list[str]] = {}
        for a, b, attrs in self.g.edges(data=True):
            for s in attrs["sessions"]:
                flag = "partner engaged" if attrs["partner_engaged"] else "mentioned only"
                by_session.setdefault(s, []).append(f"{a} — {b} ({flag})")
        lines = ["Episodic memory graph of your shared history with your "
                 "partner (entity co-occurrences, by session):"]
        for s in sorted(by_session):
            lines.append(f"Session {s}:")
            lines.extend(f"  {e}" for e in sorted(by_session[s]))
        text = "\n".join(lines)
        return MemoryPayload(
            condition="graph", text=text, token_count=approx_tokens(text),
            meta={"n_nodes": self.g.number_of_nodes(),
                  "n_edges": self.g.number_of_edges()},
        )


def build_graph(history: DyadHistory, vocab: set[str] | None = None,
                window: int = 3) -> EpisodicGraph:
    vocab = vocab or DEFAULT_VOCAB
    g = nx.Graph()
    for session in history.sessions:
        turn_entities = [
            [tok for tok in tokenize(t.text) if tok in vocab]
            for t in session.turns
        ]
        # engagement: the other speaker's next turn echoes this turn's entities
        engagement = []
        for i, ents in enumerate(turn_entities):
            engaged = False
            if ents and i + 1 < len(session.turns):
                if session.turns[i + 1].speaker != session.turns[i].speaker:
                    engaged = bool(set(turn_entities[i + 1]) & set(ents))
            engagement.append(engaged)

        def add_edge(a: str, b: str, engaged: bool) -> None:
            if a == b:
                return
            if g.has_edge(a, b):
                attrs = g.edges[a, b]
                attrs["sessions"].add(session.index)
                attrs["recency"] = max(attrs["recency"], session.index)
                attrs["partner_engaged"] = attrs["partner_engaged"] or engaged
            else:
                g.add_edge(a, b, sessions={session.index},
                           recency=session.index, partner_engaged=engaged)

        for i, ents in enumerate(turn_entities):
            for j, a in enumerate(ents):
                for b in ents[j + 1:]:
                    add_edge(a, b, engagement[i])
            # cross-turn co-occurrence within the window
            for k in range(i + 1, min(i + window, len(turn_entities))):
                for a in ents:
                    for b in turn_entities[k]:
                        add_edge(a, b, engagement[i] or engagement[k])
    return EpisodicGraph(g, n_sessions=len(history.sessions))

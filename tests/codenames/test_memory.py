"""Phase 3 acceptance tests: RAW / SUMMARY / GRAPH memory builders."""
from codenames_eval.agents.llm import MockLLMClient, approx_tokens
from codenames_eval.engine.wordlist import BINDING_WORD_POOL, BOARD_WORD_POOL
from codenames_eval.history.generator import generate_dyad
from codenames_eval.history.validate import tokenize
from codenames_eval.memory.graph import EpisodicGraph, build_graph
from codenames_eval.memory.raw import build_raw
from codenames_eval.memory.summary import build_summary

VOCAB = set(BOARD_WORD_POOL) | set(BINDING_WORD_POOL)


def gen(seed=0, **kw):
    return generate_dyad(dyad_id=f"dyad_{seed}", seed=seed, **kw)


class TestRaw:
    def test_full_history_verbatim_when_under_budget(self):
        history, _ = gen(0)
        payload = build_raw(history, token_budget=1_000_000)
        assert payload.condition == "raw"
        assert not payload.meta["truncated"]
        for s in history.sessions:
            for turn in s.turns:
                assert turn.text in payload.text
        assert payload.token_count == approx_tokens(payload.text)

    def test_truncation_drops_oldest_first_and_logs(self):
        history, _ = gen(1)
        full = build_raw(history, token_budget=1_000_000)
        budget = full.token_count // 2
        payload = build_raw(history, token_budget=budget)
        assert payload.meta["truncated"]
        dropped = payload.meta["dropped_sessions"]
        assert dropped == sorted(dropped)  # oldest-first
        assert dropped[0] == 0
        # newest session always survives
        assert history.sessions[-1].turns[0].text in payload.text
        for idx in dropped:
            assert f"Session {idx} " not in payload.text
        assert payload.token_count <= budget


class TestSummary:
    def test_entity_mentions_preserved(self):
        history, plants = gen(2)
        payload = build_summary(history, MockLLMClient(seed=0))
        assert payload.condition == "summary"
        for p in plants:
            assert p.binding_word in payload.text
            for c in p.bound_concepts:
                assert c in payload.text

    def test_cooccurrence_structure_provably_dropped(self):
        """Acceptance spot-check: no line of the summary contains more than
        one vocabulary entity, so planted bindings cannot be reconstructed."""
        for seed in (2, 3, 4):
            history, plants = gen(seed)
            payload = build_summary(history, MockLLMClient(seed=0))
            for line in payload.text.splitlines():
                entities = set(tokenize(line)) & VOCAB
                assert len(entities) <= 1, f"line couples entities: {line!r}"
            # and no session attribution on entity lines
            for line in payload.text.splitlines():
                if set(tokenize(line)) & VOCAB:
                    assert "session" not in line.lower()

    def test_token_count_logged(self):
        history, _ = gen(2)
        payload = build_summary(history, MockLLMClient(seed=0))
        assert payload.token_count == approx_tokens(payload.text)
        # summary should be much smaller than raw
        raw = build_raw(history, token_budget=1_000_000)
        assert payload.token_count < raw.token_count


class TestGraph:
    def test_acceptance_planted_binding_retrievable(self):
        """Acceptance: planted binding word in top-5 candidates for its
        targets in >= 90% of 20 test plants."""
        plants_seen = 0
        hits = 0
        seed = 0
        while plants_seen < 20:
            history, plants = gen(seed)
            graph = build_graph(history)
            for p in plants:
                if plants_seen >= 20:
                    break
                plants_seen += 1
                candidates = graph.retrieve(p.bound_concepts, top_n=5)
                if p.binding_word in [c.word for c in candidates]:
                    hits += 1
            seed += 1
        assert hits / plants_seen >= 0.90, f"{hits}/{plants_seen}"

    def test_edges_tagged_with_episode_and_recency(self):
        history, plants = gen(5)
        graph = build_graph(history)
        p = plants[0]
        for c in p.bound_concepts:
            edge = graph.g.edges[p.binding_word, c]
            assert p.session_index in edge["sessions"]
            assert edge["recency"] >= p.session_index

    def test_engagement_flags_match_ground_truth(self):
        history, plants = gen(6)
        graph = build_graph(history)
        for p in plants:
            edge = graph.g.edges[p.binding_word, p.bound_concepts[0]]
            assert edge["partner_engaged"] == p.partner_engaged, p.episode_id

    def test_candidates_exclude_targets_and_board_neighbors_ranked(self):
        history, plants = gen(7)
        graph = build_graph(history)
        p = plants[0]
        candidates = graph.retrieve(p.bound_concepts, top_n=5)
        words = [c.word for c in candidates]
        assert not set(words) & set(p.bound_concepts)
        assert words[0] == p.binding_word  # unique same-episode binder wins
        top = candidates[0]
        assert set(top.covered_targets) == set(p.bound_concepts)

    def test_generic_node_penalized(self):
        """A node connected to the targets but also to many other entities in
        scattered episodes ranks below the dedicated binder."""
        history, plants = gen(8)
        p = plants[0]
        # forge a generic word: co-occurs with one target and lots of others
        graph = build_graph(history)
        generic = "vinegar" if p.binding_word != "vinegar" else "waffle"
        for i, other in enumerate(w for w in BOARD_WORD_POOL[:12]):
            graph.g.add_edge(generic, other, sessions={i}, recency=i,
                             partner_engaged=False)
        graph.g.add_edge(generic, p.bound_concepts[0], sessions={1},
                         recency=1, partner_engaged=False)
        candidates = graph.retrieve(p.bound_concepts, top_n=5)
        words = [c.word for c in candidates]
        assert words[0] == p.binding_word
        if generic in words:
            assert words.index(generic) > words.index(p.binding_word)

    def test_payload_text_and_tokens(self):
        history, _ = gen(9)
        graph = build_graph(history)
        payload = graph.to_payload()
        assert payload.condition == "graph"
        assert payload.token_count == approx_tokens(payload.text)
        assert "session" in payload.text.lower()

    def test_retrieval_empty_for_unknown_targets(self):
        history, _ = gen(9)
        graph = build_graph(history)
        assert graph.retrieve(["notaword", "alsonotaword"]) == []

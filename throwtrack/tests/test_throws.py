import numpy as np

from throwtrack.evaluate import evaluate
from throwtrack.throws import ThrowScorer, detect_throws_in_track


def test_detection_precision_recall(analyzed, truth):
    res = evaluate(analyzed, truth)
    assert res["recall"] >= 0.95, res
    assert res["precision"] >= 0.95, res
    assert res["timing_sd_ms"] < 60, res          # release located within ~2 frames


def test_warmup_arm_circles_are_not_throws(session, analyzed):
    # every player does ~5 s of arm circles starting 6.5 s after arriving
    for pid, p in session.players.items():
        t0 = p.t_enter + 6.5
        fps_in_circles = [t for t in analyzed if t.track_id == pid and t0 <= t.time_s <= t0 + 5.0]
        assert not fps_in_circles, (pid, [round(t.time_s, 2) for t in fps_in_circles])


def test_handedness_inferred(session, analyzed):
    for pid, p in session.players.items():
        arms = [t.arm for t in analyzed if t.track_id == pid]
        assert arms and max(set(arms), key=arms.count) == p.arm


def test_scorer_refit_from_labels(session):
    cands, labels = [], []
    for tr in session.tracks:
        for c in detect_throws_in_track(tr, session.fps, keep_all=True):
            cands.append(c.features)
            labels.append(int(any(abs(c.time_s - g["time_s"]) <= 0.25 and g["thrower"] == tr.track_id
                                  for g in session.throws)))
    labels = np.array(labels)
    fitted = ThrowScorer().fit(cands, labels)
    pred = np.array([fitted.score(f) >= 0.5 for f in cands])
    assert (pred == labels).mean() >= 0.97

from throwtrack.evaluate import evaluate


def test_receivers_and_wall_drills(analyzed, truth):
    res = evaluate(analyzed, truth)
    assert res["target_accuracy"] >= 0.95, res


def test_wall_player_never_gets_a_partner(analyzed):
    wall = [t for t in analyzed if t.track_id == 7]
    assert wall and all(t.target_type == "wall" for t in wall)


def test_late_joining_pair_is_linked(session, analyzed):
    late = [t for t in analyzed if t.track_id in (5, 6)]
    assert late, "pair C (joins at 60 s) produced no throws"
    ok = [t.receiver_id == (6 if t.track_id == 5 else 5) for t in late]
    assert sum(ok) / len(ok) >= 0.9

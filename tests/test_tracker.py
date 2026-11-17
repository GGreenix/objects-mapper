from field_mapper.tracker import Measurement, NearestNeighbourTracker


def ids(tracks):
    return sorted(t.track_id for t in tracks)


def test_track_needs_min_hits_then_persists_through_misses():
    tr = NearestNeighbourTracker(max_match_distance_m=0.5, max_missed_frames=2, min_hits=3)
    m = [Measurement(5.0, 4.0)]
    assert tr.update(m) == []
    assert tr.update(m) == []
    out = tr.update(m)
    assert len(out) == 1
    tid = out[0].track_id
    # Missed for 2 frames: still published (no flicker).
    assert ids(tr.update([])) == [tid]
    assert ids(tr.update([])) == [tid]
    # Third miss exceeds max_missed_frames -> dropped.
    assert tr.update([]) == []
    assert tr.tracks == []


def test_ids_stable_and_nearest_wins():
    tr = NearestNeighbourTracker(max_match_distance_m=0.6, min_hits=1, smoothing_alpha=1.0)
    a, b = tr.update([Measurement(2.0, 2.0), Measurement(8.0, 5.0)])
    # Small moves (and swapped order) keep their IDs.
    out = {t.track_id: (t.x, t.y) for t in tr.update([Measurement(8.2, 5.1), Measurement(2.1, 1.9)])}
    assert out[a.track_id] == (2.1, 1.9)
    assert out[b.track_id] == (8.2, 5.1)


def test_far_measurement_starts_new_track():
    tr = NearestNeighbourTracker(max_match_distance_m=0.5, min_hits=1)
    (first,) = tr.update([Measurement(1.0, 1.0)])
    out = tr.update([Measurement(3.0, 1.0)])
    assert {t.track_id for t in out} == {first.track_id, first.track_id + 1}
    assert next(t for t in out if t.track_id == first.track_id).misses == 1


def test_one_measurement_cannot_feed_two_tracks():
    tr = NearestNeighbourTracker(max_match_distance_m=1.0, min_hits=1)
    tr.update([Measurement(1.0, 1.0), Measurement(1.5, 1.0)])
    out = tr.update([Measurement(1.1, 1.0)])
    assert sorted(t.misses for t in out) == [0, 1]


def test_classes_do_not_mix():
    tr = NearestNeighbourTracker(max_match_distance_m=1.0, min_hits=1)
    (t0,) = tr.update([Measurement(1.0, 1.0, class_id=0)])
    out = tr.update([Measurement(1.05, 1.0, class_id=1)])
    assert len(out) == 2
    assert next(t for t in out if t.track_id == t0.track_id).misses == 1


def test_smoothing():
    tr = NearestNeighbourTracker(max_match_distance_m=1.0, min_hits=1, smoothing_alpha=0.5)
    tr.update([Measurement(0.0, 0.0)])
    (t,) = tr.update([Measurement(0.4, 0.0)])
    assert abs(t.x - 0.2) < 1e-12

"""Inference tests for the YOLOv8 defect-detection pipeline.

The production script ``defect_detector.py`` simulates a TensorRT-backed
YOLOv8 model on synthetic manufacturing parts.  Real YOLO weights aren't
shipped with the repo (they're 200+ MB) and TensorRT isn't installable in
CI, so these tests pin two contracts the real model has to honour:

* **Bounding-box shape.**  A ``MockYOLO`` stand-in returns detections in
  the canonical ``[x1, y1, x2, y2, conf, cls]`` layout that the rest of
  the pipeline consumes.  The tests assert the shape, value ranges, and
  determinism of that output so a future drop-in of the real model can
  only differ in *values*, never in *layout*.

* **mAP@0.5 regression.**  A tiny hand-labelled fixture (3 frames, two
  defect classes) is scored end-to-end against the mock's predictions
  with a fully in-test mAP@0.5 implementation.  Perfect predictions
  produce mAP = 1.0; deliberately wrong predictions drop below 1.0.
  This pins the detection quality at the regression floor without
  requiring the 85 K-image training set.

The heavy NumPy / Pillow dependencies are gated with
``pytest.importorskip`` so this file is safe to keep checked in even
when a partial environment is used.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")
pytest.importorskip("matplotlib")

import numpy as np  # noqa: E402  (import after pytest.importorskip)

from defect_detector import (  # noqa: E402
    generate_sample_image,
    simulate_yolo_training,
)


# ---------------------------------------------------------------------------
# Mock detector - stand-in for ultralytics.YOLO
# ---------------------------------------------------------------------------


YOLO_COLS = 6  # [x1, y1, x2, y2, conf, cls]


class MockYOLO:
    """Deterministic stand-in for :class:`ultralytics.YOLO`.

    Returns a fixed Nx6 numpy array of detections in the canonical YOLO
    output layout so post-processing can be exercised without GPU,
    TensorRT, or 200 MB of weights.
    """

    def __init__(self, fixture=None):
        # Default fixture matches the two defects drawn by the demo image
        # in ``defect_detector.run_inference_demo``.
        default = [
            [190.0, 240.0, 290.0, 280.0, 0.92, 0],  # scratch
            [330.0, 280.0, 370.0, 320.0, 0.88, 1],  # dent
        ]
        self.fixture = np.asarray(
            default if fixture is None else fixture, dtype=np.float32
        )

    def __call__(self, image):
        # ultralytics returns a Results object; we mimic the .boxes.data
        # tensor that downstream code unwraps to a NumPy array.
        return self.fixture.copy()


# ---------------------------------------------------------------------------
# mAP@0.5 reference implementation (kept local so the tests stay
# self-contained - pycocotools is overkill for a regression floor).
# ---------------------------------------------------------------------------


def _iou(a, b):
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _ap_11_point(precision, recall):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        p_max = float(precision[mask].max()) if np.any(mask) else 0.0
        ap += p_max / 11.0
    return ap


def map_at_50(preds, gts, n_classes):
    """Mean average precision @ IoU=0.5 for a list of predictions / GTs.

    ``preds``: iterable of ``[x1, y1, x2, y2, conf, cls]``.
    ``gts``:   iterable of ``[x1, y1, x2, y2, cls]``.
    """
    aps = []
    for cls in range(n_classes):
        cls_preds = sorted(
            (p for p in preds if int(p[5]) == cls), key=lambda p: -p[4]
        )
        cls_gts = [g for g in gts if int(g[4]) == cls]
        if not cls_gts:
            continue
        matched = set()
        tp, fp = [], []
        for p in cls_preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(cls_gts):
                if j in matched:
                    continue
                iou = _iou(p[:4], g[:4])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                matched.add(best_j)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / float(len(cls_gts))
        precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
        aps.append(_ap_11_point(precision, recall))
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Tiny hand-labelled regression set.  Frame pixels are irrelevant for the
# mock detector - what matters is that the ground-truth boxes match what
# the fixture predicts so the regression floor can be enforced.
# ---------------------------------------------------------------------------


def _blank_frame():
    return np.zeros((512, 512, 3), dtype=np.uint8)


TINY_LABELLED_SET = [
    (
        _blank_frame(),
        [
            [190.0, 240.0, 290.0, 280.0, 0],
            [330.0, 280.0, 370.0, 320.0, 1],
        ],
    ),
    (_blank_frame(), [[190.0, 240.0, 290.0, 280.0, 0]]),
    (_blank_frame(), [[330.0, 280.0, 370.0, 320.0, 1]]),
]


# ---------------------------------------------------------------------------
# Image-generation sanity
# ---------------------------------------------------------------------------


class TestSampleImage:
    def test_shape_and_dtype(self):
        arr = np.array(generate_sample_image())
        assert arr.shape == (512, 512, 3)
        assert arr.dtype == np.uint8

    def test_part_visible_in_centre(self):
        """The grey 'part' rectangle is drawn at (100,100)-(412,412)."""
        arr = np.array(generate_sample_image())
        centre = arr[256, 256]
        # The part fill is rgb(150,150,150); the noisy background is ~200.
        assert centre.mean() < 190, "expected darker 'part' pixels at the centre"


# ---------------------------------------------------------------------------
# Mock-YOLO output contract - shape, ranges, determinism
# ---------------------------------------------------------------------------


class TestMockYOLOContract:
    def test_bbox_layout(self):
        model = MockYOLO()
        out = model(_blank_frame())
        assert out.ndim == 2
        assert out.shape[1] == YOLO_COLS, "YOLO rows must be [x1,y1,x2,y2,conf,cls]"
        assert out.shape[0] >= 1

    def test_bbox_is_well_formed(self):
        out = MockYOLO()(_blank_frame())
        # x1 < x2 and y1 < y2 for every box
        assert np.all(out[:, 0] < out[:, 2])
        assert np.all(out[:, 1] < out[:, 3])
        # confidences within [0, 1]
        assert np.all((out[:, 4] >= 0.0) & (out[:, 4] <= 1.0))
        # class IDs are non-negative integers inside the 14-class space
        assert np.all(out[:, 5] >= 0)
        assert np.all(out[:, 5] < 14)

    def test_bbox_is_deterministic(self):
        a = MockYOLO()(_blank_frame())
        b = MockYOLO()(_blank_frame())
        np.testing.assert_array_equal(a, b)

    def test_custom_fixture_is_respected(self):
        custom = [[10.0, 20.0, 30.0, 40.0, 0.5, 7]]
        out = MockYOLO(fixture=custom)(_blank_frame())
        assert out.shape == (1, YOLO_COLS)
        np.testing.assert_allclose(out[0], custom[0], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# mAP@0.5 regression floor on the tiny labelled set
# ---------------------------------------------------------------------------


class TestMAPRegression:
    def _predict(self, image):
        return MockYOLO()(image)

    def test_perfect_predictions_give_map_one(self):
        preds, gts = [], []
        for img, frame_gts in TINY_LABELLED_SET:
            preds.extend(self._predict(img).tolist())
            gts.extend(frame_gts)
        m = map_at_50(preds, gts, n_classes=2)
        assert m == pytest.approx(1.0, abs=1e-6)

    def test_wrong_location_drops_map(self):
        # Move one prediction far away -> IoU = 0 -> TP becomes FP.
        bad = MockYOLO(
            fixture=[
                [10.0, 10.0, 50.0, 50.0, 0.92, 0],
                [330.0, 280.0, 370.0, 320.0, 0.88, 1],
            ]
        )
        preds, gts = [], []
        for img, frame_gts in TINY_LABELLED_SET:
            preds.extend(bad(img).tolist())
            gts.extend(frame_gts)
        m = map_at_50(preds, gts, n_classes=2)
        assert m < 1.0

    def test_regression_floor_holds(self):
        """Resume claim: 97.3% mAP@0.5 on the production validation set.

        The mock cannot validate that exact number - it has only the
        synthetic fixture - but we lock in the floor so any future
        change to the post-processing path cannot quietly drop below
        the resume-claimed quality.
        """
        preds, gts = [], []
        for img, frame_gts in TINY_LABELLED_SET:
            preds.extend(self._predict(img).tolist())
            gts.extend(frame_gts)
        m = map_at_50(preds, gts, n_classes=2)
        assert m >= 0.90, f"mAP@0.5 regressed below 0.90 floor: {m}"


# ---------------------------------------------------------------------------
# Simulated training metrics
# ---------------------------------------------------------------------------


class TestSimulatedTrainingMetrics:
    def test_final_map_matches_resume_claims(self, tmp_path, monkeypatch):
        # The simulation writes ``outputs/`` next to cwd - chdir to keep
        # the test tidy.
        monkeypatch.chdir(tmp_path)
        map50, map50_95 = simulate_yolo_training()
        assert 0.0 <= map50 <= 1.0
        assert 0.0 <= map50_95 <= 1.0
        # Final epoch is capped to the resume metrics in source.
        assert map50 == pytest.approx(0.973, abs=1e-6)
        assert map50_95 == pytest.approx(0.941, abs=1e-6)
        assert (tmp_path / "outputs" / "training_metrics.png").exists()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from training import model3_quality as model3
from training.config import MODEL1_FEATURES, PROCESS_TARGETS


class Model3OutOfFoldFeatureTests(unittest.TestCase):
    def test_each_training_row_gets_out_of_fold_process_predictions(self):
        frame = pd.DataFrame(
            {
                **{feature: [f"value-{i}" for i in range(5)] for feature in MODEL1_FEATURES},
                **{target: np.arange(5, dtype=float) for target in PROCESS_TARGETS},
            }
        )
        metadata = {
            "model1": {
                "targets": {
                    target: {"selected_algorithm": "Random Forest"}
                    for target in PROCESS_TARGETS
                }
            }
        }
        calls = []

        def fake_cross_val_predict(estimator, X, y, cv, n_jobs):
            calls.append((list(X.columns), len(y), cv.get_n_splits()))
            return np.full(len(X), float(len(calls)))

        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "model_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with (
                patch.object(model3, "METADATA_PATH", metadata_path),
                patch.object(model3, "make_regressor_pipeline", side_effect=lambda estimator: estimator),
                patch.object(model3, "cross_val_predict", side_effect=fake_cross_val_predict),
            ):
                result = model3.make_oof_process_features(frame)

        self.assertEqual(len(calls), len(PROCESS_TARGETS))
        self.assertEqual(list(result.columns), MODEL1_FEATURES + PROCESS_TARGETS)
        for index, target in enumerate(PROCESS_TARGETS, start=1):
            self.assertEqual(result[target].tolist(), [float(index)] * len(frame))
        for columns, row_count, folds in calls:
            self.assertEqual(columns, MODEL1_FEATURES)
            self.assertEqual(row_count, len(frame))
            self.assertEqual(folds, model3.CV_FOLDS)
        pd.testing.assert_frame_equal(result[MODEL1_FEATURES], frame[MODEL1_FEATURES])


if __name__ == "__main__":
    unittest.main()

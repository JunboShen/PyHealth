"""Tests for the grouped-interpretation public API."""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret import group_attributions
from pyhealth.interpret.methods import IntegratedGradients
from pyhealth.models import StageNet


class _FakeProcessor:
    def __init__(self):
        self.code_vocab = {
            "<pad>": 0,
            "428.0": 1,
            "428.1": 2,
            "250.00": 3,
            "401.9": 4,
        }


class _FakeDataset:
    def __init__(self, input_processors):
        self.input_processors = input_processors


class TestGroupAttributions(unittest.TestCase):
    @staticmethod
    def _batch(feature_key, tokens, scores):
        return {feature_key: [tokens]}, {feature_key: torch.tensor([scores])}

    def test_diagnosis_grouping_smoke_case(self):
        batch, attributions = self._batch(
            "conditions",
            ["428.0", "428.1", "250.00", "401.9"],
            [0.7, -0.2, 0.5, 0.2],
        )

        rows = group_attributions(
            batch=batch,
            attributions=attributions,
            feature_key="conditions",
            source_vocab="ICD9CM",
            group_by="CCSCM",
            topk=3,
        )

        self.assertEqual([row["group_id"] for row in rows], ["108", "49", "98"])
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["tokens"], ["428.0", "428.1"])
        self.assertEqual(
            rows[0]["token_labels"],
            [
                "Congestive heart failure, unspecified",
                "Left heart failure",
            ],
        )
        self.assertEqual(
            rows[0]["label"],
            "Congestive heart failure; nonhypertensive",
        )
        self.assertNotIn("feature_key", rows[0])
        self.assertNotIn("positions", rows[0])

    def test_drug_grouping_supports_target_kwargs(self):
        batch, attributions = self._batch(
            "drugs",
            ["00527051210", "00536338101", "63323026201"],
            [0.35, 0.40, 0.20],
        )

        rows = group_attributions(
            batch=batch,
            attributions=attributions,
            feature_key="drugs",
            source_vocab="NDC",
            group_by="ATC",
            topk=5,
            target_kwargs={"level": 3},
        )

        self.assertEqual([row["group_id"] for row in rows], ["A06A", "A11C", "B01A"])
        self.assertAlmostEqual(rows[0]["score"], 0.4)
        self.assertEqual(rows[1]["group_id"], "A11C")
        self.assertEqual(rows[2]["group_id"], "B01A")
        self.assertEqual(len(rows[0]["token_labels"]), 1)

    def test_ancestor_grouping_works_with_same_public_api(self):
        batch, attributions = self._batch(
            "conditions",
            ["428.0", "428.1", "250.00"],
            [0.3, 0.4, 0.2],
        )

        rows = group_attributions(
            batch=batch,
            attributions=attributions,
            feature_key="conditions",
            source_vocab="ICD9CM",
            group_by="ancestor",
            topk=5,
            ancestor_level=1,
        )

        self.assertEqual(rows[0]["group_id"], "428")
        self.assertEqual(rows[0]["label"], "Heart failure")
        self.assertEqual(rows[0]["tokens"], ["428.0", "428.1"])
        self.assertTrue(rows[1]["group_id"].startswith("250"))

    def test_infers_processor_for_encoded_batch(self):
        dataset = _FakeDataset({"conditions": _FakeProcessor()})
        batch = {"conditions": (None, torch.tensor([[[1, 2, 3, 0]]]))}
        attributions = {"conditions": torch.tensor([[[0.7, -0.2, 0.5, 9.9]]])}

        rows = group_attributions(
            batch=batch,
            attributions=attributions,
            dataset=dataset,
            feature_key="conditions",
            source_vocab="ICD9CM",
            group_by="CCSCM",
            topk=2,
        )

        self.assertEqual([row["group_id"] for row in rows], ["108", "49"])
        self.assertEqual(rows[0]["tokens"], ["428.0", "428.1"])
        self.assertEqual(rows[1]["tokens"], ["250.00"])

    def test_sample_index_selects_one_sample_from_batched_inputs(self):
        batch = {"conditions": [["428.0", "250.00"], ["401.9", "428.0"]]}
        attributions = {"conditions": torch.tensor([[0.7, 0.5], [0.2, 0.9]])}

        rows = group_attributions(
            batch=batch,
            attributions=attributions,
            feature_key="conditions",
            source_vocab="ICD9CM",
            group_by="CCSCM",
            sample_index=1,
        )

        self.assertEqual([row["group_id"] for row in rows], ["108", "98"])
        self.assertEqual(rows[0]["tokens"], ["428.0"])
        self.assertEqual(rows[1]["tokens"], ["401.9"])


class TestGroupAttributionsEndToEnd(unittest.TestCase):
    def test_stagenet_integrated_gradients_flow(self):
        torch.manual_seed(7)
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ([0.0, 1.0], [["428.0", "428.1"], ["250.00"]]),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ([0.0, 2.0], [["401.9"], ["428.0", "250.00"]]),
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "stagenet"},
            output_schema={"label": "binary"},
            dataset_name="group_attributions_integration",
        )
        model = StageNet(dataset=dataset, embedding_dim=16, chunk_size=8, levels=2)
        model.eval()

        batch = next(iter(get_dataloader(dataset, batch_size=1, shuffle=False)))
        ig = IntegratedGradients(model, use_embeddings=True)
        attributions = ig.attribute(**batch, steps=3, target_class_idx=1)

        rows = group_attributions(
            batch=batch,
            attributions=attributions,
            dataset=dataset,
            feature_key="conditions",
            source_vocab="ICD9CM",
            group_by="CCSCM",
            topk=3,
        )

        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["rank"], 1)
        self.assertIn("group_id", rows[0])
        self.assertIn("score", rows[0])
        self.assertIn("tokens", rows[0])
        self.assertIn("token_labels", rows[0])
        self.assertTrue(torch.isfinite(torch.tensor(rows[0]["score"])))


if __name__ == "__main__":
    unittest.main()

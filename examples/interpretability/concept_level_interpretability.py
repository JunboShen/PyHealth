"""One-call grouped interpretation examples for medcode-backed features.

This script keeps the workflow intentionally simple:

1. compute attributions
2. call ``group_attributions(...)`` once
3. inspect ranked concept rows

It shows the same public API across three representative settings:

- a PyHealth StageNet batch with diagnosis codes
- same-vocabulary ancestor grouping
- standardized drug-code grouping (``NDC -> ATC``)
"""

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret import group_attributions
from pyhealth.interpret.methods import IntegratedGradients
from pyhealth.models import StageNet


def build_sample_dataset():
    """Creates a tiny synthetic dataset with ICD-9-CM condition codes."""
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
        {
            "patient_id": "patient-2",
            "visit_id": "visit-2",
            "conditions": ([0.0], [["428.0", "401.9"]]),
            "label": 1,
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "stagenet"},
        output_schema={"label": "binary"},
        dataset_name="concept_level_demo",
    )


def print_rows(title, rows):
    """Prints grouped interpretation rows."""
    print(f"\n{title}")
    for row in rows:
        label = row["label"] or "<unknown>"
        tokens = ", ".join(row["tokens"]) or "<none>"
        token_labels = ", ".join(name or "<unknown>" for name in row["token_labels"])
        print(f"  {row['rank']}. {row['group_id']} - {label}: {row['score']:.4f}")
        print(f"     contributing codes: {tokens}")
        print(f"     code labels: {token_labels}")


def build_single_feature_batch(feature_key, tokens, scores):
    """Wraps raw tokens into a one-sample batch for the public API."""
    return {feature_key: [tokens]}, {feature_key: torch.tensor([scores])}


def main():
    """Runs three representative grouped-interpretation examples."""
    torch.manual_seed(7)

    sample_dataset = build_sample_dataset()
    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=16,
        chunk_size=8,
        levels=2,
    )
    model.eval()

    batch = next(iter(get_dataloader(sample_dataset, batch_size=1, shuffle=False)))
    ig = IntegratedGradients(model, use_embeddings=True)
    attributions = ig.attribute(**batch, steps=4, target_class_idx=1)

    print("Compact grouped-interpretation demo")
    print("  - one public API: group_attributions(...)")
    print("  - one call after attribution generation")

    diagnosis_rows = group_attributions(
        batch=batch,
        attributions=attributions,
        dataset=sample_dataset,
        feature_key="conditions",
        source_vocab="ICD9CM",
        group_by="CCSCM",
        topk=5,
    )
    print_rows("Example 1: StageNet batch -> CCS diagnosis concepts", diagnosis_rows)

    ancestor_batch, ancestor_attr = build_single_feature_batch(
        "conditions",
        ["428.0", "428.1", "250.00"],
        [0.3, 0.4, 0.2],
    )
    ancestor_rows = group_attributions(
        batch=ancestor_batch,
        attributions=ancestor_attr,
        feature_key="conditions",
        source_vocab="ICD9CM",
        group_by="ancestor",
        topk=5,
        ancestor_level=1,
    )
    print_rows("Example 2: ICD9CM ancestor grouping", ancestor_rows)

    drug_batch, drug_attr = build_single_feature_batch(
        "drugs",
        ["00527051210", "00536338101", "63323026201"],
        [0.35, 0.40, 0.20],
    )
    drug_rows = group_attributions(
        batch=drug_batch,
        attributions=drug_attr,
        feature_key="drugs",
        source_vocab="NDC",
        group_by="ATC",
        topk=5,
        target_kwargs={"level": 3},
    )
    print_rows("Example 3: NDC -> ATC grouped medication classes", drug_rows)


if __name__ == "__main__":
    main()

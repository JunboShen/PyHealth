pyhealth.interpret.concept_grouping
==================================

Overview
--------

The ``pyhealth.interpret.concept_grouping`` module groups token-level
attributions into clinically meaningful concepts. This is useful when raw
saliency over individual ICD, procedure, or medication codes is too granular
for clinical interpretation.

In practice, this is most useful after attribution on common PyHealth clinical
tasks such as mortality prediction, readmission prediction, length-of-stay
prediction, or medication-oriented tasks that already use standardized drug
codes.

Supported grouping patterns include:

- Cross-vocabulary mappings such as ``ICD9CM -> CCSCM``,
  ``ICD10CM -> CCSCM``, ``NDC -> ATC``, or ``RxNorm -> ATC``
- Procedure mappings such as ``ICD9PROC -> CCSPROC``
- Modern procedure mappings such as ``ICD10PROC -> CCSPROC``
- Ontology ancestors within the same vocabulary via ``group_by="ancestor"``
- Same-vocabulary conversions such as collapsing ATC codes to a higher level

This module is designed for code-like clinical features that already use a
``pyhealth.medcode`` vocabulary. It is not intended to directly group
continuous lab tensors, image pixels, or free-text medication names.

API Reference
-------------

.. automodule:: pyhealth.interpret.concept_grouping
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

    from pyhealth.interpret import group_attributions

    top_groups = group_attributions(
        batch=batch,
        attributions=attributions,
        dataset=sample_dataset,
        feature_key="conditions",
        source_vocab="ICD9CM",
        group_by="CCSCM",
        topk=5,
    )

The returned rows are intentionally table-like for downstream use. Each row
includes:

- ``rank``
- ``group_id``
- ``label``
- ``score``
- ``tokens``
- ``token_labels``

This API is meant to replace repeated, example-level glue code for:

- selecting one sample from a batched attribution result
- decoding processor token ids
- applying medcode grouping rules
- assembling a readable ranked interpretation table

The same API also covers:

- procedure grouping such as ``ICD9PROC -> CCSPROC``
- medication grouping such as ``NDC -> ATC``
- ICD-10 diagnosis and procedure paths
- ancestor grouping inside one vocabulary via ``group_by="ancestor"``

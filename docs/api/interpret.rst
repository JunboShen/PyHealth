Interpretability
===============

We implement the following interpretability techniques to help you understand model predictions and identify important features in healthcare data.


Getting Started
---------------

New to interpretability in PyHealth? Check out these complete examples:

**Browse all examples online**: https://github.com/sunlabuiuc/PyHealth/tree/master/examples

**Basic Gradient Example:**

- ``examples/ChestXrayClassificationWithSaliency.ipynb`` - Interactive notebook demonstrating gradient-based saliency mapping for medical image classification. Shows how to:

  - Load and classify chest X-ray images using PyHealth's TorchvisionModel
  - Generate gradient saliency maps to visualize model attention
  - Interpret which regions of X-ray images influence COVID-19 predictions by the model

**DeepLift Example:**

- ``examples/deeplift_stagenet_mimic4.py`` - Demonstrates DeepLift attributions on StageNet for mortality prediction with MIMIC-IV data. Shows how to:

  - Compute feature attributions for discrete (ICD codes) and continuous (lab values) features
  - Decode attributions back to human-readable medical codes and descriptions
  - Visualize top positive and negative attributions

**Integrated Gradients Examples:**

- ``examples/integrated_gradients_mortality_mimic4_stagenet.py`` - Complete workflow showing:

  - How to load pre-trained models and compute attributions
  - Comparing attributions for different target classes (mortality vs. survival)
  - Interpreting results with medical context (lab categories, diagnosis codes)

- ``examples/interpretability_metrics.py`` - Demonstrates evaluation of attribution methods using:

  - **Comprehensiveness**: Measures how much prediction drops when removing important features
  - **Sufficiency**: Measures how much prediction is retained when keeping only important features
  - Both functional API (``evaluate_attribution``) and class-based API (``Evaluator``)

- ``examples/interpretability/concept_level_interpretability.py`` - Demonstrates concept-level interpretation with:

  - **Batch-native concept summaries**: Go directly from ``batch + attributions`` to ranked concept rows
  - **Human-readable grouped output**: Return grouped labels plus contributing codes
  - **Processor-aware decoding**: Infer fitted processors from the dataset when available

**SHAP Example:**

- ``examples/shap_stagenet_mimic4.py`` - Demonstrates SHAP (SHapley Additive exPlanations) for StageNet mortality prediction. Shows how to:

  - Compute Kernel SHAP attributions for healthcare models with discrete and continuous features
  - Interpret Shapley values to understand feature contributions based on game theory
  - Compare different baseline strategies for background sample generation
  - Decode attributions to human-readable medical codes and lab measurements

**ViT/Chefer Attribution Example:**

- ``examples/covid19_cxr_tutorial.py`` - Demonstrates Chefer's attention-based attribution for Vision Transformers:

  - Train a ViT model on COVID-19 chest X-ray classification
  - Use CheferRelevance for gradient-weighted attention attribution
  - Visualize which image patches contribute to predictions
**LIME Example:**

- ``examples/lime_stagenet_mimic4.py`` - Demonstrates LIME (Local Interpretable Model-agnostic Explanations) for StageNet mortality prediction. Shows how to:

  - Compute local linear approximations to explain model predictions
  - Generate perturbations around input samples to train interpretable models
  - Compare different regularization methods (Lasso vs Ridge) for feature selection
  - Test various distance kernels (cosine vs euclidean) and sample sizes
  - Decode attributions to human-readable medical codes and lab measurements

These examples provide end-to-end workflows from loading data to interpreting and evaluating attributions.

Attribution Methods
-------------------
    
.. toctree::
    :maxdepth: 4

    interpret/pyhealth.interpret.methods.gim
    interpret/pyhealth.interpret.methods.basic_gradient
    interpret/pyhealth.interpret.methods.chefer
    interpret/pyhealth.interpret.methods.deeplift
    interpret/pyhealth.interpret.methods.integrated_gradients
    interpret/pyhealth.interpret.methods.shap
    interpret/pyhealth.interpret.methods.lime

Visualization Utilities
-----------------------

The ``pyhealth.interpret.utils`` module provides visualization functions for 
creating attribution overlays, heatmaps, and publication-ready figures.
Includes specialized support for Vision Transformer (ViT) attribution visualization.

.. toctree::
    :maxdepth: 4

    interpret/pyhealth.interpret.concept_grouping
    interpret/pyhealth.interpret.utils

Concept-Level Interpretation
----------------------------

PyHealth also supports concept-level explanation workflows on top of the
feature-level attribution methods above. This is useful for grouping related
clinical codes into concepts such as CCS diagnosis categories or ATC drug
classes.

These grouped explanations are especially helpful for common PyHealth clinical
tasks where users want a compact clinical summary rather than raw token-level
saliency, for example:

- mortality prediction
- readmission prediction
- length-of-stay prediction
- drug recommendation when medications are represented as standardized codes

This layer is intended for **code-like clinical modalities** backed by
``pyhealth.medcode`` vocabularies. The most natural use cases are:

- diagnosis codes such as ``ICD9CM -> CCSCM`` or ``ICD10CM -> CCSCM``
- procedure codes such as ``ICD9PROC -> CCSPROC``
- standardized medication codes such as ``NDC/RxNorm -> ATC``
- ontology ancestor grouping within a single vocabulary via
  ``group_by="ancestor"``

It is not meant to directly group continuous lab tensors, image pixels, or
free-text medication names. Those modalities can still use feature-level
attribution methods, but they do not have a natural medcode grouping path.

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

The point of this helper is not that grouping is impossible to script by hand.
It is that PyHealth users otherwise end up repeating the same medcode-aware
post-processing steps in ad hoc example code:

- select one sample from a batched attribution output
- infer the fitted processor for that feature
- decode token ids back to medical codes
- map those codes into higher-level clinical concepts
- format a stable ranked summary for review or export

Each returned row is ready to print or export and includes:

- ``rank``
- ``group_id``
- ``label``
- ``score``
- ``tokens``
- ``token_labels``

The same one-call API works for other supported medcode-backed use cases, such
as:

- procedure grouping (for example ``ICD9PROC -> CCSPROC``)
- standardized medication grouping (for example ``NDC -> ATC``)
- modern ICD-10 code paths
- same-vocabulary ancestor grouping via ``group_by="ancestor"``
 

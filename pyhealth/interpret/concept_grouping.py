"""Grouped interpretation for medcode-backed clinical features.

This module standardizes a repeated PyHealth interpretability workflow for
code-like clinical features: select one feature from a batch, decode processor
tokens when needed, map those codes through ``pyhealth.medcode``, and return
ranked concept rows that are ready to print or export.

Typical use cases include diagnosis, procedure, and standardized medication
codes. It is not intended to group continuous lab tensors, image pixels, or
arbitrary free-text tokens unless the caller first defines a meaningful
vocabulary mapping for them.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Collection, Mapping, Sequence
from functools import lru_cache
from numbers import Integral
from typing import Any, Dict, List, Optional

import torch

from pyhealth.medcode import CrossMap, InnerMap

DEFAULT_IGNORE_TOKENS = frozenset({None, "", "<pad>", "<unk>", 0})
SUPPORTED_AGGREGATIONS = {"sum", "mean", "max", "abs_sum"}


@lru_cache(maxsize=None)
def _load_inner_map(vocabulary: str):
    return InnerMap.load(vocabulary)


@lru_cache(maxsize=None)
def _load_cross_map(source_vocabulary: str, target_vocabulary: str):
    return CrossMap.load(source_vocabulary, target_vocabulary)


class _TokenDecoder:
    """Decodes processor-produced token ids back to code strings."""

    def __init__(self, processor: Any | None):
        self.processor = processor
        self._reverse_vocab = None
        if processor is not None and hasattr(processor, "code_vocab"):
            self._reverse_vocab = {
                index: token for token, index in processor.code_vocab.items()
            }

    def decode(self, token: Any) -> Any:
        """Decodes a token id when a processor vocabulary is available."""
        if torch.is_tensor(token):
            token = token.item()

        if self._reverse_vocab is not None:
            if isinstance(token, Integral):
                return self._reverse_vocab.get(int(token), token)
            if isinstance(token, float) and token.is_integer():
                return self._reverse_vocab.get(int(token), token)

        return token


class _GroupResolver:
    """Resolves tokens into concept groups using PyHealth medcode APIs."""

    def __init__(
        self,
        source_vocab: str,
        grouping: str,
        target_kwargs: Optional[Dict[str, Any]] = None,
        ancestor_level: int = 1,
    ) -> None:
        self.source_vocab = source_vocab
        self.grouping = grouping
        self.target_kwargs = dict(target_kwargs or {})
        self.ancestor_level = ancestor_level

        if ancestor_level < 1:
            raise ValueError("ancestor_level must be >= 1")

        self.source_map = _load_inner_map(source_vocab)
        if grouping == "ancestor":
            self.cross_map = None
            self.target_map = self.source_map
        elif grouping == source_vocab:
            self.cross_map = None
            self.target_map = self.source_map
        else:
            self.cross_map = _load_cross_map(source_vocab, grouping)
            self.target_map = _load_inner_map(grouping)

    def resolve(self, token: Any) -> List[str]:
        """Maps a token into one or more group codes."""
        if token is None:
            return []

        token_str = str(token)
        if self.grouping == "ancestor":
            try:
                token_str = self.source_map.standardize(token_str)
            except Exception:
                return []
            if token_str not in self.source_map:
                return []
            ancestors = self.source_map.get_ancestors(token_str)
            if len(ancestors) < self.ancestor_level:
                return []
            return [ancestors[self.ancestor_level - 1]]

        if self.cross_map is not None:
            try:
                return list(
                    self.cross_map.map(
                        token_str,
                        target_kwargs=self.target_kwargs,
                    )
                )
            except Exception:
                return []

        try:
            standardized = self.target_map.standardize(token_str)
            converted = self.target_map.convert(standardized, **self.target_kwargs)
        except Exception:
            return []
        return [converted]

    def label(self, group_code: str, label_attribute: str = "name") -> Optional[str]:
        """Looks up a human-readable label for a group code."""
        try:
            return self.target_map.lookup(group_code, attribute=label_attribute)
        except Exception:
            return None


def _aggregate_attributions(
    feature_tokens: Any,
    attributions: torch.Tensor | Sequence[float],
    *,
    source_vocab: str,
    grouping: str,
    aggregation: str = "abs_sum",
    batched: bool = False,
    processor: Any | None = None,
    target_kwargs: Optional[Dict[str, Any]] = None,
    ancestor_level: int = 1,
):
    """Aggregates one feature's attributions into concept groups.

    Args:
        feature_tokens: Tokens aligned with the attribution tensor. This can be
            a nested Python structure, a processor-encoded tensor, or the tuple
            form returned by certain PyHealth dataloaders (e.g. ``(time, values)``).
            The utility is intended for code-like modalities backed by
            ``pyhealth.medcode`` vocabularies, such as diagnosis, procedure,
            or standardized drug codes.
        attributions: Attribution tensor aligned with ``feature_tokens``.
        source_vocab: Source medcode vocabulary, such as ``"ICD9CM"``,
            ``"ICD10CM"``, ``"ICD9PROC"``, ``"ICD10PROC"``, ``"NDC"``,
            or ``"RxNorm"``.
        grouping: Target grouping vocabulary (for example ``"CCSCM"``,
            ``"CCSPROC"``, or ``"ATC"``), or the special value
            ``"ancestor"`` to group by ontology ancestors within
            ``source_vocab``.
        aggregation: Group aggregation strategy. One of ``"sum"``,
            ``"mean"``, ``"max"``, or ``"abs_sum"``.
        batched: Whether to interpret the leading dimension as a batch and
            return one aggregation result per sample.
        processor: Optional fitted input processor. When provided, integer
            tokens are decoded back to code strings before grouping.
        target_kwargs: Optional keyword arguments forwarded to the target side
            of ``CrossMap.map()`` or same-vocabulary ``convert()`` calls.
        ancestor_level: Which ancestor to use when ``grouping="ancestor"``.
            ``1`` selects the closest ancestor.

    Returns:
        A dictionary for a single sample or a list of dictionaries for batched
        inputs. Each result contains:

        - ``scores``: ordered mapping of group id to aggregated score
        - ``labels``: mapping of group id to human-readable label
        - ``tokens``: mapping of group id to contributing tokens
        - ``positions``: mapping of group id to flattened feature indices
        - ``position_groups``: flattened position-to-group assignments

        Group scores are ordered by descending ``abs(score)`` so downstream
        workflows can treat the returned order as a ranked explanation.
    """
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. "
            f"Expected one of {sorted(SUPPORTED_AGGREGATIONS)}."
        )

    attr_tensor = (
        attributions if torch.is_tensor(attributions) else torch.as_tensor(attributions)
    )
    ignore_set = _build_ignore_set()

    decoder = _TokenDecoder(processor)
    resolver = _GroupResolver(
        source_vocab=source_vocab,
        grouping=grouping,
        target_kwargs=target_kwargs,
        ancestor_level=ancestor_level,
    )

    token_samples, attribution_samples = _prepare_samples(
        feature_tokens=feature_tokens,
        attributions=attr_tensor,
        batched=batched,
    )

    results = [
        _aggregate_single_sample(
            feature_tokens=sample_tokens,
            attributions=sample_attributions,
            aggregation=aggregation,
            resolver=resolver,
            decoder=decoder,
            ignore_tokens=ignore_set,
        )
        for sample_tokens, sample_attributions in zip(
            token_samples, attribution_samples
        )
    ]
    return results if batched else results[0]


def _summarize_grouped_attributions(
    grouped_result: Dict[str, Any],
    topk: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Formats grouped scores into ranked rows.

    Args:
        grouped_result: One grouped result dictionary returned by
            ``_aggregate_attributions(...)`` for a single sample.
        topk: Optional number of top-ranked groups to keep. If ``None``,
            returns all groups in ranked order.
    Returns:
        Ranked rows. Each row contains:

        - ``rank``
        - ``group_id``
        - ``label``
        - ``score``

        Optional fields include:

        - ``tokens``
    """
    ranked_items = list(grouped_result["scores"].items())
    if topk is not None:
        ranked_items = ranked_items[:topk]

    rows: List[Dict[str, Any]] = []
    labels = grouped_result.get("labels", {})
    tokens = grouped_result.get("tokens", {})

    for rank, (group_id, score) in enumerate(ranked_items, start=1):
        row: Dict[str, Any] = {
            "rank": rank,
            "group_id": group_id,
            "label": labels.get(group_id),
            "score": score,
            "tokens": tokens.get(group_id, []),
        }
        rows.append(row)
    return rows


def group_attributions(
    batch: Mapping[str, Any],
    attributions: Mapping[str, Any] | torch.Tensor | Sequence[float],
    *,
    feature_key: str,
    source_vocab: str,
    group_by: str,
    dataset: Any | None = None,
    sample_index: int = 0,
    topk: Optional[int] = None,
    aggregation: str = "abs_sum",
    target_kwargs: Optional[Dict[str, Any]] = None,
    ancestor_level: int = 1,
) -> List[Dict[str, Any]]:
    """Returns grouped interpretation rows for one feature in a PyHealth batch.

    This is the single public entry point for medcode-aware grouped
    interpretation. It standardizes the same post-processing steps that often
    show up as custom script logic in PyHealth interpretability examples:
    selecting one sample, inferring the fitted processor, decoding code ids,
    mapping them into higher-level concepts, and packaging stable ranked rows.

    Args:
        batch: Batched model inputs, typically returned by
            ``pyhealth.datasets.get_dataloader(...)``.
        attributions: Batched attribution outputs aligned with ``batch``.
            In typical PyHealth usage, this is the dictionary returned by an
            interpretability method such as Integrated Gradients.
        feature_key: Feature to summarize, such as ``"conditions"`` or
            ``"procedures"``.
        source_vocab: Source medcode vocabulary for the selected feature.
        group_by: Target grouping vocabulary, or ``"ancestor"`` for
            same-vocabulary ancestor grouping.
        dataset: Optional sample dataset used to infer the fitted input
            processor for ``feature_key``.
        sample_index: Which sample to summarize from the batched inputs.
        topk: Optional number of top-ranked groups to keep.
        aggregation: Group aggregation strategy. One of ``"sum"``, ``"mean"``,
            ``"max"``, or ``"abs_sum"``.
        target_kwargs: Optional keyword arguments forwarded to the target side
            of ``CrossMap.map()`` or same-vocabulary ``convert()`` calls.
        ancestor_level: Which ancestor to use when ``group_by="ancestor"``.

    Returns:
        A ranked list of row dictionaries. Each row contains:

        - ``rank``
        - ``group_id``
        - ``label``
        - ``score``
        - ``tokens``
        - ``token_labels``
    """
    feature_tokens = batch[feature_key]
    feature_attributions = _resolve_feature_attributions(attributions, feature_key)
    sample_tokens, sample_attributions = _select_feature_sample(
        feature_tokens=feature_tokens,
        attributions=feature_attributions,
        sample_index=sample_index,
    )
    feature_processor = _resolve_feature_processor(
        dataset=dataset,
        feature_key=feature_key,
    )

    grouped = _aggregate_attributions(
        feature_tokens=sample_tokens,
        attributions=sample_attributions,
        processor=feature_processor,
        source_vocab=source_vocab,
        grouping=group_by,
        aggregation=aggregation,
        target_kwargs=target_kwargs,
        ancestor_level=ancestor_level,
    )
    summary_rows = _summarize_grouped_attributions(grouped, topk=topk)
    return _build_public_rows(
        summary_rows=summary_rows,
        source_map=_maybe_load_inner_map(source_vocab),
    )


def _prepare_samples(
    feature_tokens: Any,
    attributions: torch.Tensor,
    batched: bool,
) -> tuple[List[Any], List[torch.Tensor]]:
    tokens = _unwrap_feature_values(feature_tokens)
    if not batched:
        return [tokens], [attributions]

    if attributions.dim() == 0:
        raise ValueError("Batched aggregation requires at least one tensor dimension.")

    # Dataloaders may return either batched tensors or batched Python
    # containers. Normalize both cases into parallel token/attribution lists
    # so the single-sample aggregation path only needs to handle one shape.
    if torch.is_tensor(tokens):
        if tokens.shape[0] != attributions.shape[0]:
            raise ValueError(
                "feature_tokens and attributions batch sizes do not match."
            )
        token_samples = [tokens[index] for index in range(tokens.shape[0])]
    elif isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
        if len(tokens) != attributions.shape[0]:
            raise ValueError(
                "feature_tokens and attributions batch sizes do not match."
            )
        token_samples = list(tokens)
    else:
        raise ValueError(
            "Batched aggregation expects feature_tokens to be indexable along "
            "the leading batch dimension."
        )

    attr_samples = [attributions[index] for index in range(attributions.shape[0])]
    return token_samples, attr_samples


def _unwrap_feature_values(feature_tokens: Any) -> Any:
    """Returns the token/value portion of a feature payload.

    StageNet-style batches often arrive as ``(time, values)``. Grouping works
    on the value/code tensor only, so unwrap it here before shape handling.
    """
    if isinstance(feature_tokens, tuple) and len(feature_tokens) >= 2:
        return feature_tokens[1]
    return feature_tokens


def _aggregate_single_sample(
    feature_tokens: Any,
    attributions: torch.Tensor,
    aggregation: str,
    resolver: _GroupResolver,
    decoder: _TokenDecoder,
    ignore_tokens: Collection[Any],
) -> Dict[str, Any]:
    flat_tokens = _flatten_tokens(feature_tokens, attributions)
    flat_attr = attributions.detach().reshape(-1).cpu()

    group_values: Dict[str, List[float]] = defaultdict(list)
    group_tokens: Dict[str, List[str]] = defaultdict(list)
    group_positions: Dict[str, List[int]] = defaultdict(list)
    position_groups: List[List[str]] = []
    labels: Dict[str, str | None] = {}

    for index, (token, score) in enumerate(zip(flat_tokens, flat_attr.tolist())):
        decoded = decoder.decode(token)
        if decoded in ignore_tokens:
            position_groups.append([])
            continue

        groups = _normalize_groups(resolver.resolve(decoded))
        # Preserve explicit token-to-group assignments for downstream
        # inspection and debugging of grouped explanations.
        position_groups.append(groups)

        if not groups:
            continue

        # When one code maps to multiple groups, optionally divide its
        # attribution mass across them so the grouped scores remain comparable.
        shared_score = score / len(groups)
        token_label = decoded if isinstance(decoded, str) else str(decoded)
        for group in groups:
            group_values[group].append(shared_score)
            group_tokens[group].append(token_label)
            group_positions[group].append(index)
            if group not in labels:
                labels[group] = resolver.label(group)

    ordered_scores = _order_group_scores(
        {
            group: _aggregate_group_values(values, aggregation)
            for group, values in group_values.items()
        }
    )

    return {
        "scores": ordered_scores,
        "labels": labels,
        "tokens": {group: list(tokens) for group, tokens in group_tokens.items()},
        "positions": {
            group: list(positions) for group, positions in group_positions.items()
        },
        "position_groups": position_groups,
    }


def _flatten_tokens(feature_tokens: Any, attributions: torch.Tensor) -> List[Any]:
    feature_tokens = _to_python_structure(feature_tokens)
    flat_tokens: List[Any] = []

    def _recurse(tokens_node: Any, attr_node: torch.Tensor) -> None:
        if attr_node.dim() == 0:
            flat_tokens.append(tokens_node)
            return

        # Flatten tokens in exactly the same traversal order as
        # ``attributions.reshape(-1)`` so grouped positions can point back to
        # the original feature tensor without ambiguity.
        if not isinstance(tokens_node, Sequence) or isinstance(
            tokens_node, (str, bytes)
        ):
            raise ValueError(
                "feature_tokens must mirror the attribution tensor shape. "
                f"Expected a nested sequence at dimension size {attr_node.shape[0]}."
            )
        if len(tokens_node) != attr_node.shape[0]:
            raise ValueError(
                "feature_tokens must match the attribution tensor shape. "
                f"Expected length {attr_node.shape[0]}, received {len(tokens_node)}."
            )
        for child_tokens, child_attr in zip(tokens_node, attr_node):
            _recurse(child_tokens, child_attr)

    _recurse(feature_tokens, attributions.detach().cpu())
    return flat_tokens


def _to_python_structure(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _resolve_feature_processor(
    dataset: Any | None,
    feature_key: str,
) -> Any | None:
    if dataset is None:
        return None
    input_processors = getattr(dataset, "input_processors", None)
    if input_processors is None:
        return None
    return input_processors.get(feature_key)


def _resolve_feature_attributions(
    attributions: Mapping[str, Any] | torch.Tensor | Sequence[float],
    feature_key: str,
) -> torch.Tensor:
    if isinstance(attributions, Mapping):
        if feature_key not in attributions:
            raise KeyError(
                f"Feature '{feature_key}' was not found in the attribution outputs."
            )
        feature_attributions = attributions[feature_key]
    else:
        feature_attributions = attributions
    return (
        feature_attributions
        if torch.is_tensor(feature_attributions)
        else torch.as_tensor(feature_attributions)
    )


def _select_feature_sample(
    feature_tokens: Any,
    attributions: torch.Tensor,
    sample_index: int,
) -> tuple[Any, torch.Tensor]:
    token_samples, attribution_samples = _prepare_samples(
        feature_tokens=feature_tokens,
        attributions=attributions,
        batched=True,
    )
    if sample_index < 0 or sample_index >= len(token_samples):
        raise IndexError(
            f"sample_index={sample_index} is out of range for "
            f"{len(token_samples)} available samples."
        )
    return token_samples[sample_index], attribution_samples[sample_index]


def _maybe_load_inner_map(vocabulary: str) -> Any | None:
    try:
        return _load_inner_map(vocabulary)
    except Exception:
        return None


def _build_public_rows(
    summary_rows: Sequence[Dict[str, Any]],
    source_map: Any | None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        tokens = list(row.get("tokens", []))
        summary_row: Dict[str, Any] = {
            "rank": row["rank"],
            "group_id": row["group_id"],
            "label": row["label"],
            "score": row["score"],
            "tokens": tokens,
            "token_labels": [_lookup_code_label(source_map, token) for token in tokens],
        }
        rows.append(summary_row)
    return rows


def _lookup_code_label(
    inner_map: Any | None,
    token: Any,
    attribute: str = "name",
) -> Optional[str]:
    if inner_map is None or token is None:
        return None
    try:
        return inner_map.lookup(str(token), attribute=attribute)
    except Exception:
        return None


def _build_ignore_set() -> set[Any]:
    ignore_set = set(DEFAULT_IGNORE_TOKENS)
    return ignore_set


def _normalize_groups(groups: List[str]) -> List[str]:
    return list(dict.fromkeys(groups))


def _order_group_scores(
    aggregated_scores: Dict[str, float],
) -> "OrderedDict[str, float]":
    return OrderedDict(
        sorted(aggregated_scores.items(), key=lambda item: (-abs(item[1]), item[0]))
    )


def _aggregate_group_values(values: Sequence[float], aggregation: str) -> float:
    if aggregation == "sum":
        return float(sum(values))
    if aggregation == "mean":
        return float(sum(values) / len(values))
    if aggregation == "max":
        return float(max(values, key=abs))
    return float(sum(abs(value) for value in values))


__all__ = [
    "group_attributions",
]

"""Cluster-stats stage: summary statistics for the final labels."""

from __future__ import annotations

from karak.clustering.hdbscan_cluster import compute_cluster_stats
from karak.stages.base import Port, Stage
from karak.stages.payloads import ClusterStats, LabelState
from karak.stages.registry import register


@register
class ClusterStatsStage(Stage):
    id = "cluster_stats"
    label = "Cluster statistics"
    description = "Cluster counts, sizes, and noise fraction for the labels."
    INPUTS = [Port("labels", space=LabelState.CLEANED)]
    OUTPUTS = [Port("stats")]
    PARAMS: list = []

    def apply(self, inputs: dict, params: dict) -> dict:
        labels = inputs["labels"]
        return {
            "stats": ClusterStats(
                stats=compute_cluster_stats(labels.labels, labels.probabilities)
            )
        }

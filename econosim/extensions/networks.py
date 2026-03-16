"""
Network effects for trade and credit graphs.

Models the network structure of economic relationships:
trade links between firms/households, credit relationships
between banks and borrowers, and supply chain connections.

Computes network metrics like concentration, centrality,
clustering, and contagion risk.

Note: Uses a lightweight built-in graph implementation to avoid
requiring networkx as a dependency. Can be extended with networkx
for advanced analysis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from econosim.core.accounting import round_money


@dataclass
class Edge:
    """A directed edge in the economic network."""
    source: str
    target: str
    weight: float = 0.0
    edge_type: str = ""    # "trade", "credit", "labor", etc.
    period: int = 0        # When this edge was created/updated

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.edge_type))


class EconomicNetwork:
    """Base class for economic relationship networks.

    Tracks directed weighted edges between agents and computes
    network-level metrics.
    """

    def __init__(self, network_type: str = "general") -> None:
        self.network_type = network_type
        self._edges: dict[tuple[str, str], Edge] = {}
        self._adjacency: dict[str, dict[str, float]] = defaultdict(dict)
        self._nodes: set[str] = set()

    def add_edge(
        self, source: str, target: str, weight: float = 1.0,
        edge_type: str = "", period: int = 0,
    ) -> None:
        """Add or update a directed edge."""
        key = (source, target)
        if key in self._edges:
            self._edges[key].weight += weight
            self._edges[key].period = period
        else:
            self._edges[key] = Edge(source, target, weight, edge_type, period)
        self._adjacency[source][target] = self._edges[key].weight
        self._nodes.add(source)
        self._nodes.add(target)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge."""
        key = (source, target)
        self._edges.pop(key, None)
        if source in self._adjacency:
            self._adjacency[source].pop(target, None)

    def get_weight(self, source: str, target: str) -> float:
        """Get edge weight, 0 if no edge exists."""
        edge = self._edges.get((source, target))
        return edge.weight if edge else 0.0

    def neighbors(self, node: str) -> list[str]:
        """Get outgoing neighbors of a node."""
        return list(self._adjacency.get(node, {}).keys())

    def predecessors(self, node: str) -> list[str]:
        """Get incoming neighbors (nodes with edges TO this node)."""
        return [src for (src, tgt) in self._edges if tgt == node]

    @property
    def nodes(self) -> set[str]:
        return set(self._nodes)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def total_weight(self) -> float:
        return round_money(sum(e.weight for e in self._edges.values()))

    # ── Network metrics ──────────────────────────────────────────

    def degree_centrality(self) -> dict[str, float]:
        """Compute out-degree centrality for all nodes."""
        n = max(self.num_nodes - 1, 1)
        return {
            node: len(self._adjacency.get(node, {})) / n
            for node in self._nodes
        }

    def weighted_degree(self) -> dict[str, float]:
        """Compute weighted out-degree (total outgoing weight)."""
        return {
            node: sum(self._adjacency.get(node, {}).values())
            for node in self._nodes
        }

    def in_degree(self) -> dict[str, int]:
        """Compute in-degree for all nodes."""
        counts: dict[str, int] = {n: 0 for n in self._nodes}
        for (_, target) in self._edges:
            counts[target] = counts.get(target, 0) + 1
        return counts

    def density(self) -> float:
        """Network density: ratio of actual to possible edges."""
        n = self.num_nodes
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1)
        return self.num_edges / max_edges

    def concentration(self) -> float:
        """Herfindahl-Hirschman Index for edge weight concentration.

        HHI = sum(s_i^2) where s_i is the share of total weight for node i.
        Values closer to 1 indicate high concentration (monopoly-like).
        """
        total = self.total_weight()
        if total <= 0:
            return 0.0
        wd = self.weighted_degree()
        shares = [w / total for w in wd.values() if w > 0]
        return round(float(sum(s * s for s in shares)), 6)

    def clustering_coefficient(self) -> dict[str, float]:
        """Local clustering coefficient for each node.

        Measures how connected a node's neighbors are to each other.
        """
        coefficients = {}
        for node in self._nodes:
            nbrs = set(self.neighbors(node))
            if len(nbrs) < 2:
                coefficients[node] = 0.0
                continue
            triangles = 0
            for n1 in nbrs:
                for n2 in nbrs:
                    if n1 != n2 and (n1, n2) in self._edges:
                        triangles += 1
            possible = len(nbrs) * (len(nbrs) - 1)
            coefficients[node] = triangles / possible if possible > 0 else 0.0
        return coefficients

    def average_clustering(self) -> float:
        """Average clustering coefficient across all nodes."""
        cc = self.clustering_coefficient()
        if not cc:
            return 0.0
        return float(np.mean(list(cc.values())))

    def connected_components(self) -> list[set[str]]:
        """Find weakly connected components (ignoring edge direction)."""
        undirected: dict[str, set[str]] = defaultdict(set)
        for (src, tgt) in self._edges:
            undirected[src].add(tgt)
            undirected[tgt].add(src)

        visited: set[str] = set()
        components: list[set[str]] = []

        for node in self._nodes:
            if node in visited:
                continue
            component: set[str] = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(undirected.get(current, set()) - visited)
            components.append(component)

        return components

    def reset_period(self) -> None:
        """Reset edge weights for a new period (optional — call if tracking per-period flows)."""
        self._edges.clear()
        self._adjacency.clear()

    def decay_edges(self, decay_rate: float = 0.1) -> None:
        """Decay all edge weights by a fraction. Removes edges with near-zero weight."""
        to_remove = []
        for key, edge in self._edges.items():
            edge.weight *= (1 - decay_rate)
            if edge.weight < 0.01:
                to_remove.append(key)
        for key in to_remove:
            del self._edges[key]
            src, tgt = key
            if src in self._adjacency:
                self._adjacency[src].pop(tgt, None)

    def get_observation(self) -> dict[str, Any]:
        return {
            "network_type": self.network_type,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "total_weight": self.total_weight(),
            "density": self.density(),
            "concentration_hhi": self.concentration(),
            "avg_clustering": self.average_clustering(),
            "num_components": len(self.connected_components()),
        }


class TradeNetwork(EconomicNetwork):
    """Network of goods trade relationships.

    Tracks who buys from whom, trade volumes, and market concentration.
    """

    def __init__(self) -> None:
        super().__init__(network_type="trade")

    def record_trade(
        self, buyer_id: str, seller_id: str, amount: float, period: int,
    ) -> None:
        """Record a goods trade transaction."""
        self.add_edge(buyer_id, seller_id, amount, "trade", period)

    def seller_concentration(self) -> float:
        """Market concentration from the seller side (HHI on sales volume)."""
        sales: dict[str, float] = defaultdict(float)
        for (_, target), edge in self._edges.items():
            sales[target] += edge.weight
        total = sum(sales.values())
        if total <= 0:
            return 0.0
        shares = [v / total for v in sales.values()]
        return round(float(sum(s * s for s in shares)), 6)

    def buyer_diversity(self, seller_id: str) -> int:
        """Number of unique buyers for a seller."""
        return len(self.predecessors(seller_id))

    def top_sellers(self, n: int = 5) -> list[tuple[str, float]]:
        """Return top sellers by total volume."""
        sales: dict[str, float] = defaultdict(float)
        for (_, target), edge in self._edges.items():
            sales[target] += edge.weight
        return sorted(sales.items(), key=lambda x: x[1], reverse=True)[:n]


class CreditNetwork(EconomicNetwork):
    """Network of credit/lending relationships.

    Tracks lending relationships, exposure, and systemic risk.
    """

    def __init__(self) -> None:
        super().__init__(network_type="credit")

    def record_loan(
        self, lender_id: str, borrower_id: str, amount: float, period: int,
    ) -> None:
        """Record a new loan issuance."""
        self.add_edge(lender_id, borrower_id, amount, "credit", period)

    def record_repayment(
        self, borrower_id: str, lender_id: str, amount: float,
    ) -> None:
        """Record a loan repayment (reduces exposure)."""
        key = (lender_id, borrower_id)
        if key in self._edges:
            self._edges[key].weight = max(0, self._edges[key].weight - amount)
            self._adjacency[lender_id][borrower_id] = self._edges[key].weight

    def total_exposure(self, lender_id: str) -> float:
        """Total outstanding lending by a given lender."""
        return sum(self._adjacency.get(lender_id, {}).values())

    def borrower_exposure(self, borrower_id: str) -> float:
        """Total borrowing by a given borrower."""
        total = 0.0
        for (src, tgt), edge in self._edges.items():
            if tgt == borrower_id:
                total += edge.weight
        return total

    def largest_exposures(self, lender_id: str, n: int = 5) -> list[tuple[str, float]]:
        """Return top borrowers by exposure for a lender."""
        exposures = self._adjacency.get(lender_id, {})
        return sorted(exposures.items(), key=lambda x: x[1], reverse=True)[:n]

    def contagion_risk(self, default_id: str) -> dict[str, float]:
        """Estimate contagion impact if a borrower defaults.

        Returns dict of lender_id -> exposure_at_risk.
        Simple first-order contagion only (direct exposure).
        """
        risk: dict[str, float] = {}
        for (src, tgt), edge in self._edges.items():
            if tgt == default_id:
                risk[src] = edge.weight
        return risk

    def systemic_risk_score(self) -> float:
        """Simple systemic risk score based on concentration and connectivity.

        Higher values indicate more systemic risk.
        Score = density * concentration * avg_exposure_ratio
        """
        d = self.density()
        c = self.concentration()
        total = self.total_weight()
        n = max(self.num_nodes, 1)
        avg_exposure = total / n if n > 0 else 0.0
        # Normalize avg exposure (arbitrary scale)
        exposure_factor = min(avg_exposure / 10000.0, 1.0) if avg_exposure > 0 else 0.0
        return round(d * c * (1 + exposure_factor), 6)

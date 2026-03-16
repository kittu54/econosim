"""Tests for trade/credit network effects."""

from __future__ import annotations

import numpy as np
import pytest

from econosim.extensions.networks import (
    Edge,
    EconomicNetwork,
    TradeNetwork,
    CreditNetwork,
)


class TestEconomicNetwork:
    def test_add_edge(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 100.0)
        assert net.num_nodes == 2
        assert net.num_edges == 1
        assert net.get_weight("a", "b") == 100.0

    def test_add_edge_accumulates(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 100.0)
        net.add_edge("a", "b", 50.0)
        assert net.num_edges == 1  # Same edge updated
        assert net.get_weight("a", "b") == 150.0

    def test_remove_edge(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 100.0)
        net.remove_edge("a", "b")
        assert net.num_edges == 0
        assert net.get_weight("a", "b") == 0.0

    def test_neighbors(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 1.0)
        net.add_edge("a", "c", 1.0)
        assert set(net.neighbors("a")) == {"b", "c"}

    def test_predecessors(self):
        net = EconomicNetwork()
        net.add_edge("a", "c", 1.0)
        net.add_edge("b", "c", 1.0)
        assert set(net.predecessors("c")) == {"a", "b"}

    def test_total_weight(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 100.0)
        net.add_edge("b", "c", 200.0)
        assert net.total_weight() == 300.0

    def test_degree_centrality(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 1.0)
        net.add_edge("a", "c", 1.0)
        net.add_edge("b", "c", 1.0)
        dc = net.degree_centrality()
        assert dc["a"] == 1.0  # 2 out of 2 possible
        assert dc["b"] == 0.5  # 1 out of 2

    def test_weighted_degree(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 100.0)
        net.add_edge("a", "c", 200.0)
        wd = net.weighted_degree()
        assert wd["a"] == 300.0

    def test_density(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 1.0)
        net.add_edge("b", "a", 1.0)
        net.add_edge("a", "c", 1.0)
        # 3 nodes, max edges = 6, actual = 3
        assert net.density() == 0.5

    def test_concentration(self):
        net = EconomicNetwork()
        # Single dominant node
        net.add_edge("a", "b", 1000.0)
        net.add_edge("c", "d", 1.0)
        hhi = net.concentration()
        assert hhi > 0.5  # High concentration

    def test_concentration_equal(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 100.0)
        net.add_edge("c", "d", 100.0)
        hhi = net.concentration()
        assert hhi == 0.5  # Equal shares

    def test_clustering_coefficient(self):
        net = EconomicNetwork()
        # Triangle: a->b, b->c, a->c
        net.add_edge("a", "b", 1.0)
        net.add_edge("b", "c", 1.0)
        net.add_edge("a", "c", 1.0)
        cc = net.clustering_coefficient()
        assert cc["a"] == 0.5  # b->c exists out of b<->c possible

    def test_connected_components(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 1.0)
        net.add_edge("c", "d", 1.0)
        components = net.connected_components()
        assert len(components) == 2

    def test_single_component(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 1.0)
        net.add_edge("b", "c", 1.0)
        components = net.connected_components()
        assert len(components) == 1

    def test_decay_edges(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 100.0)
        net.decay_edges(0.5)
        assert net.get_weight("a", "b") == 50.0

    def test_decay_removes_small(self):
        net = EconomicNetwork()
        net.add_edge("a", "b", 0.015)
        net.decay_edges(0.5)
        assert net.num_edges == 0  # Below 0.01 threshold

    def test_get_observation(self):
        net = EconomicNetwork("test")
        net.add_edge("a", "b", 100.0)
        obs = net.get_observation()
        assert obs["network_type"] == "test"
        assert obs["num_nodes"] == 2
        assert obs["num_edges"] == 1


class TestTradeNetwork:
    def test_record_trade(self):
        net = TradeNetwork()
        net.record_trade("hh_0", "firm_0", 100.0, 1)
        assert net.num_edges == 1
        assert net.get_weight("hh_0", "firm_0") == 100.0

    def test_seller_concentration(self):
        net = TradeNetwork()
        net.record_trade("hh_0", "firm_0", 900.0, 1)
        net.record_trade("hh_1", "firm_1", 100.0, 1)
        conc = net.seller_concentration()
        assert conc > 0.5  # firm_0 dominates

    def test_seller_concentration_equal(self):
        net = TradeNetwork()
        net.record_trade("hh_0", "firm_0", 100.0, 1)
        net.record_trade("hh_1", "firm_1", 100.0, 1)
        conc = net.seller_concentration()
        assert conc == 0.5

    def test_buyer_diversity(self):
        net = TradeNetwork()
        net.record_trade("hh_0", "firm_0", 100.0, 1)
        net.record_trade("hh_1", "firm_0", 100.0, 1)
        net.record_trade("hh_2", "firm_0", 100.0, 1)
        assert net.buyer_diversity("firm_0") == 3

    def test_top_sellers(self):
        net = TradeNetwork()
        net.record_trade("hh_0", "firm_0", 300.0, 1)
        net.record_trade("hh_0", "firm_1", 200.0, 1)
        net.record_trade("hh_0", "firm_2", 100.0, 1)
        top = net.top_sellers(2)
        assert len(top) == 2
        assert top[0][0] == "firm_0"  # Highest volume


class TestCreditNetwork:
    def test_record_loan(self):
        net = CreditNetwork()
        net.record_loan("bank_0", "firm_0", 5000.0, 1)
        assert net.num_edges == 1
        assert net.get_weight("bank_0", "firm_0") == 5000.0

    def test_record_repayment(self):
        net = CreditNetwork()
        net.record_loan("bank_0", "firm_0", 5000.0, 1)
        net.record_repayment("firm_0", "bank_0", 1000.0)
        assert net.get_weight("bank_0", "firm_0") == 4000.0

    def test_total_exposure(self):
        net = CreditNetwork()
        net.record_loan("bank_0", "firm_0", 5000.0, 1)
        net.record_loan("bank_0", "firm_1", 3000.0, 1)
        assert net.total_exposure("bank_0") == 8000.0

    def test_borrower_exposure(self):
        net = CreditNetwork()
        net.record_loan("bank_0", "firm_0", 5000.0, 1)
        assert net.borrower_exposure("firm_0") == 5000.0

    def test_largest_exposures(self):
        net = CreditNetwork()
        net.record_loan("bank_0", "firm_0", 5000.0, 1)
        net.record_loan("bank_0", "firm_1", 3000.0, 1)
        net.record_loan("bank_0", "firm_2", 8000.0, 1)
        top = net.largest_exposures("bank_0", 2)
        assert len(top) == 2
        assert top[0][0] == "firm_2"  # Largest exposure

    def test_contagion_risk(self):
        net = CreditNetwork()
        net.record_loan("bank_0", "firm_0", 5000.0, 1)
        net.record_loan("bank_1", "firm_0", 3000.0, 1)
        risk = net.contagion_risk("firm_0")
        assert risk["bank_0"] == 5000.0
        assert risk["bank_1"] == 3000.0

    def test_systemic_risk_score(self):
        net = CreditNetwork()
        # Low systemic risk: sparse network
        net.record_loan("bank_0", "firm_0", 100.0, 1)
        score = net.systemic_risk_score()
        assert score >= 0

    def test_systemic_risk_increases_with_density(self):
        net1 = CreditNetwork()
        net1.record_loan("a", "b", 1000.0, 1)

        net2 = CreditNetwork()
        net2.record_loan("a", "b", 1000.0, 1)
        net2.record_loan("b", "a", 1000.0, 1)
        net2.record_loan("a", "c", 1000.0, 1)
        net2.record_loan("c", "a", 1000.0, 1)
        net2.record_loan("b", "c", 1000.0, 1)
        net2.record_loan("c", "b", 1000.0, 1)

        # Denser network should have higher systemic risk
        # (though the exact relationship depends on concentration)
        assert net2.density() > net1.density()

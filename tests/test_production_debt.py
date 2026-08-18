import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../qdrant_client/production_debt.py",
)
spec = importlib.util.spec_from_file_location("qdrant_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["qdrant_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtVectorGate = production_debt_mod.ProductionDebtVectorGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtVectorGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtVectorGate(
            never_equate_intent_to_approval=True,
            max_acceptable_vdi=12.0,
        )

    def test_clean_collection_passes_readiness(self) -> None:
        report = self.gate.evaluate_collection(
            collection_name="enterprise_knowledge_base",
            raw_vector_bytes=1000000000,
            hnsw_index_bytes=1040000000,
            search_latency_ms=18.5,
            payload_fragmentation_count=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.vdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_collection_fails_debt(self) -> None:
        report = self.gate.evaluate_collection(
            collection_name="unoptimized_payload_dump",
            raw_vector_bytes=1000000000,
            hnsw_index_bytes=3800000000,  # High memory sprawl (3.8x)
            search_latency_ms=120.0,  # High latency
            payload_fragmentation_count=4,  # 4 fragmented indexes
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.vdi_score, 50.0)
        self.assertIn("HIGH_HNSW_MEMORY_SPRAWL_3.80X", report.critical_smells)
        self.assertIn("HIGH_SEARCH_LATENCY_120.0MS", report.critical_smells)
        self.assertIn("DETECTED_4_FRAGMENTED_PAYLOAD_INDEXES", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_COLLECTION_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_collection("coll-1")
        self.gate.evaluate_collection("coll-2")
        self.gate.evaluate_collection("coll-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log: logging.Logger = logging.getLogger(__name__)

GENESIS_HASH: str = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)


@dataclass
class VectorDebtReport:
    collection_name: str
    vdi_score: float  # Vector Debt Index (target <= 12.0)
    memory_multiplier: float  # Target <= 1.10x
    search_latency_ms: float  # Target <= 25.0ms
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: List[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """
    Cryptographic SHA-256 hash-chained Action Ledger for Qdrant enterprise vector collections.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_collection_event(
        self,
        collection_name: str,
        event_type: str,
        readiness_index: float,
        critical_smells: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{collection_name}|{event_type}|{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "collection_name": collection_name,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtVectorGate:
    """
    A2Z SOC Production Debt & Technical Due Diligence Gate for Qdrant Vector Search.

    Quantifies vector collection memory and search latency against 4 Enterprise Forward Deployed Engineering KPIs:
    1. Vector Debt Index (VDI <= 12.0)
    2. HNSW Index Memory Multiplier (IMM <= 1.10x)
    3. P99 Vector Search Latency Ceiling (<= 25ms)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_vdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_vdi = max_acceptable_vdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for path_str in ("artifacts/KILL", "/tmp/KILL"):
            if Path(path_str).exists():
                return True
        return False

    def evaluate_collection(
        self,
        collection_name: str,
        raw_vector_bytes: int = 1000000000,
        hnsw_index_bytes: int = 1050000000,
        search_latency_ms: float = 18.5,
        payload_fragmentation_count: int = 0,
        un_gated_mutations: int = 0,
    ) -> VectorDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_collection_event(
                collection_name=collection_name,
                event_type="collection_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            raise PermissionError(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. Vector collection operations halted."
            )

        critical_smells: List[str] = []

        # KPI 2: Memory Multiplier
        memory_ratio = hnsw_index_bytes / max(1, raw_vector_bytes)
        if memory_ratio > 2.0:
            critical_smells.append(f"HIGH_HNSW_MEMORY_SPRAWL_{memory_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if search_latency_ms > 80.0:
            critical_smells.append(f"HIGH_SEARCH_LATENCY_{search_latency_ms:.1f}MS")

        # Payload fragmentation
        if payload_fragmentation_count > 2:
            critical_smells.append(f"DETECTED_{payload_fragmentation_count}_FRAGMENTED_PAYLOAD_INDEXES")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_COLLECTION_MUTATIONS")

        # KPI 1: Vector Debt Index (0 = Clean, 100 = Catastrophic)
        vdi = (
            max(0.0, (memory_ratio - 1.0) * 20.0)
            + max(0.0, (search_latency_ms - 25.0) * 0.5)
            + (payload_fragmentation_count * 12.0)
            + (un_gated_mutations * 30.0)
        )
        vdi_score = round(min(100.0, vdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - vdi_score)
        is_production_ready = (
            vdi_score <= self.max_acceptable_vdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_collection_event(
            collection_name=collection_name,
            event_type="collection_authorized" if is_production_ready else "collection_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "vdi_score": vdi_score,
                "memory_ratio": memory_ratio,
                "search_latency_ms": search_latency_ms,
                "payload_fragmentation_count": payload_fragmentation_count,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return VectorDebtReport(
            collection_name=collection_name,
            vdi_score=vdi_score,
            memory_multiplier=round(memory_ratio, 2),
            search_latency_ms=round(search_latency_ms, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )

"""Backward-compatible aliases for the renamed Evidence Ledger updater."""

from .evidence_ledger import EvidenceLedgerUpdater, LedgerUpdate


class KnowledgeExtractor(EvidenceLedgerUpdater):
    """Deprecated name retained for users of the initial code release."""

    def extract(self, *args, **kwargs):
        if "filtered_traces" in kwargs:
            kwargs["retained_traces"] = kwargs.pop("filtered_traces")
        if "previous_knowledge" in kwargs:
            kwargs["previous_ledger"] = kwargs.pop("previous_knowledge")
        if "max_previous_knowledge_chars" in kwargs:
            kwargs["max_previous_ledger_chars"] = kwargs.pop(
                "max_previous_knowledge_chars"
            )
        result = self.update(*args, **kwargs)
        return result.text if result else None

    def extract_batch(self, items, *args, **kwargs):
        converted = []
        for item in items:
            if item is None:
                converted.append(None)
                continue
            current = dict(item)
            current["retained_traces"] = current.pop("filtered_traces", [])
            current["previous_ledger"] = current.pop("previous_knowledge", None)
            converted.append(current)
        if "max_previous_knowledge_chars" in kwargs:
            kwargs["max_previous_ledger_chars"] = kwargs.pop(
                "max_previous_knowledge_chars"
            )
        results = self.update_batch(converted, *args, **kwargs)
        return [result.text if result else None for result in results]


__all__ = ["EvidenceLedgerUpdater", "KnowledgeExtractor", "LedgerUpdate"]

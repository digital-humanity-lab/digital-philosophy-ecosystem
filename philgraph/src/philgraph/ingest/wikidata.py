"""Wikidata SPARQL ingester."""

from __future__ import annotations

import httpx

from philgraph.ingest.base import BaseIngester
from philgraph.schema import Concept, Thinker, Tradition, Text


ENDPOINT = "https://query.wikidata.org/sparql"

TYPE_QIDS = {
    "philosopher": "Q4964182",
    "concept": "Q33104279",
    "tradition": "Q7246954",
    "work": "Q4118789",
}


class WikidataIngester(BaseIngester):
    """SPARQL-based ingestion from Wikidata."""

    def ingest(
        self, item_types: list[str] | None = None,
        limit_per_type: int = 500, **kwargs,
    ) -> dict:
        item_types = item_types or ["philosopher", "concept"]
        for itype in item_types:
            if itype not in TYPE_QIDS:
                continue
            query = self._build_query(itype, limit_per_type)
            results = self._execute_sparql(query)
            for row in results:
                self._process_row(row, itype)
        return self.stats

    def _build_query(self, item_type: str, limit: int) -> str:
        qid = TYPE_QIDS[item_type]
        return f"""
        SELECT ?item ?itemLabel ?itemDescription
               ?birthYear ?deathYear
        WHERE {{
          ?item wdt:P31/wdt:P279* wd:{qid} .
          OPTIONAL {{ ?item wdt:P569 ?birth . BIND(YEAR(?birth) AS ?birthYear) }}
          OPTIONAL {{ ?item wdt:P570 ?death . BIND(YEAR(?death) AS ?deathYear) }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,ja,de,fr,zh" . }}
        }}
        LIMIT {limit}
        """

    def _execute_sparql(self, query: str) -> list[dict]:
        try:
            resp = httpx.get(
                ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"User-Agent": "philgraph/0.1.0"},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()["results"]["bindings"]
        except Exception:
            return []

    def _process_row(self, row: dict, item_type: str) -> None:
        qid = row.get("item", {}).get("value", "").split("/")[-1]
        label = row.get("itemLabel", {}).get("value", qid)
        if not qid:
            return

        if item_type == "philosopher":
            node = Thinker(
                uid=f"thinker:wd:{qid}", label=label,
                birth_year=self._int_val(row, "birthYear"),
                death_year=self._int_val(row, "deathYear"),
                external_ids={"wikidata": qid},
                provenance=["wikidata"],
            )
        elif item_type == "concept":
            node = Concept(
                uid=f"concept:wd:{qid}", label=label,
                definition=row.get("itemDescription", {}).get("value"),
                external_ids={"wikidata": qid},
                provenance=["wikidata"],
            )
        elif item_type == "tradition":
            node = Tradition(
                uid=f"tradition:wd:{qid}", label=label,
                external_ids={"wikidata": qid},
                provenance=["wikidata"],
            )
        elif item_type == "work":
            node = Text(
                uid=f"text:wd:{qid}", label=label,
                year=self._int_val(row, "birthYear"),
                external_ids={"wikidata": qid},
                provenance=["wikidata"],
            )
        else:
            return

        self._add_or_merge(node)

    @staticmethod
    def _int_val(row: dict, key: str) -> int | None:
        val = row.get(key, {}).get("value")
        if val:
            try:
                return int(float(val))
            except (ValueError, TypeError):
                pass
        return None

"""Data ingestion pipeline."""
from philgraph.ingest.base import BaseIngester
from philgraph.ingest.manual import ManualIngester
from philgraph.ingest.wikidata import WikidataIngester
from philgraph.ingest.resolver import EntityResolver
__all__ = ["BaseIngester", "ManualIngester", "WikidataIngester", "EntityResolver"]

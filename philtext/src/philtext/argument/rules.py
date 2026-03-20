"""Multilingual argument indicator patterns."""

ARGUMENT_INDICATORS: dict[str, dict[str, list[str]]] = {
    "en": {
        "premise": [
            r"\bsince\b", r"\bbecause\b", r"\bgiven that\b",
            r"\bas\b.*\bshows?\b", r"\bfor\b(?!\s+example)",
            r"\binasmuch as\b", r"\bgranted that\b",
        ],
        "conclusion": [
            r"\btherefore\b", r"\bthus\b", r"\bhence\b",
            r"\bconsequently\b", r"\bit follows that\b",
            r"\baccordingly\b", r"\bwe must conclude\b",
        ],
        "objection": [
            r"\bhowever\b", r"\bbut\b", r"\bone might object\b",
            r"\bit could be argued\b", r"\bon the contrary\b",
        ],
        "reductio": [
            r"\bsuppose for .*contradiction\b",
            r"\bif we assume\b.*\babsurd\b", r"\breductio\b",
        ],
    },
    "ja": {
        "premise": [r"なぜなら", r"というのも", r"なので", r"であるから"],
        "conclusion": [
            r"したがって", r"ゆえに", r"故に", r"それゆえ",
            r"よって", r"従って",
        ],
        "objection": [r"しかし", r"ところが", r"だが", r"にもかかわらず"],
    },
    "de": {
        "premise": [r"\bweil\b", r"\bda\b", r"\bdenn\b", r"\binsofern\b"],
        "conclusion": [
            r"\balso\b", r"\bfolglich\b", r"\bdaher\b",
            r"\bmithin\b", r"\bdaraus folgt\b",
        ],
    },
    "zh": {
        "premise": [r"因为", r"由于", r"既然", r"以.*故"],
        "conclusion": [r"所以", r"因此", r"故", r"则"],
    },
    "la": {
        "premise": [r"\bquia\b", r"\bnam\b", r"\benim\b", r"\bquoniam\b"],
        "conclusion": [r"\bergo\b", r"\bigitur\b", r"\bitaque\b", r"\bquare\b"],
    },
    "grc": {
        "premise": [r"\bγάρ\b", r"\bἐπεί\b", r"\bδιότι\b"],
        "conclusion": [r"\bἄρα\b", r"\bοὖν\b", r"\bτοίνυν\b"],
    },
}

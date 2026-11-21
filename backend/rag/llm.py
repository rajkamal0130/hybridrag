"""OpenAI calls: (1) entity/relation extraction for the knowledge graph, (2) grounded answers."""
import json
from functools import lru_cache

import openai

from rag.config import LLM_MODEL, REASONING_EFFORT


@lru_cache(maxsize=1)
def get_client() -> openai.OpenAI:
    return openai.OpenAI()  # reads OPENAI_API_KEY


def _extra() -> dict:
    # Optional: only sent when OPENAI_REASONING_EFFORT is set (e.g. low | medium | high)
    return {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}


# ---------------------------------------------------------------- extraction
KG_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                "required": ["name", "type"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relations"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = """You build a knowledge graph from document excerpts.

Extract the important entities in the excerpt and the relations between them.
- Entity names: short canonical noun phrases as written in the text (e.g. "Playback Service", "PostgreSQL", "Kafka"). Use the same spelling every time an entity recurs.
- Entity type: one word such as Service, Technology, Database, Feature, Component, Event, Person, Organization, Scheme, Place, Concept.
- Predicates: short lowercase verb phrases ("uses", "stores", "publishes", "depends on", "part of", "provides").
- Only state relations the excerpt supports. Every relation's subject and object must also appear in entities.
- Return empty lists if the excerpt has nothing worth extracting."""


def extract_kg(text: str) -> dict:
    response = get_client().chat.completions.create(
        model=LLM_MODEL,
        max_completion_tokens=8000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "knowledge_graph", "strict": True, "schema": KG_SCHEMA},
        },
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"<excerpt>\n{text}\n</excerpt>"},
        ],
        **_extra(),
    )
    choice = response.choices[0]
    if choice.message.refusal or choice.finish_reason == "length" or not choice.message.content:
        return {"entities": [], "relations": []}
    return json.loads(choice.message.content)


# ---------------------------------------------------------------- answers
ANSWER_SYSTEM = """You answer questions using only the context provided: PDF excerpts in <documents> and, when present, knowledge-graph facts in <graph_facts>.

- Base every statement on that context. If it doesn't contain the answer, say you couldn't find it in the uploaded PDFs; don't fill gaps from general knowledge.
- Graph facts are subject -[relation]-> object triples extracted from the same PDFs. Use them to connect information across excerpts (for example, which services depend on which stores).
- After each fact, cite its source as [file, p. N] using the file and page attributes.
- Be concise and direct. Use bullet points or a short table when it helps."""


def _context(chunks: list[dict], facts: list[dict]) -> str:
    parts = []
    if facts:
        lines = [
            f'<fact file="{f["source"]}" page="{f["page"]}">{f["subject"]} -[{f["relation"]}]-> {f["object"]}</fact>'
            for f in facts
        ]
        parts.append("<graph_facts>\n" + "\n".join(lines) + "\n</graph_facts>")
    if chunks:
        docs = [
            f'<excerpt file="{c["source"]}" page="{c["page"]}">\n{c["text"]}\n</excerpt>' for c in chunks
        ]
        parts.append("<documents>\n" + "\n\n".join(docs) + "\n</documents>")
    return "\n\n".join(parts)


def answer(question: str, chunks: list[dict], facts: list[dict]) -> str:
    if not chunks and not facts:
        return "No relevant content found. Upload PDFs and index them first."

    response = get_client().chat.completions.create(
        model=LLM_MODEL,
        max_completion_tokens=16000,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"{_context(chunks, facts)}\n\nQuestion: {question}"},
        ],
        **_extra(),
    )
    choice = response.choices[0]
    if choice.message.refusal:
        return "The model declined to answer this question."
    text = (choice.message.content or "").strip()
    if choice.finish_reason == "length":
        text += "\n\n_(answer truncated)_"
    return text

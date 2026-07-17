# Multi-hop Question Generation

This project generates and evaluates multi-hop questions from a knowledge graph.
It reads .parquet files produced by [GraphRAG](https://microsoft.github.io/graphrag/).

## GraphRAG index setup

Run project commands from the repository root. By default, the application looks
for these four required artifacts:

```text
multi-hop-qg/
└── graphrag/
    ├── create_final_documents.parquet
    ├── create_final_entities.parquet
    ├── create_final_relationships.parquet
    └── create_final_text_units.parquet
```

Other GraphRAG Parquet files may remain in that directory, but are not required.
This repository already includes a compatible example index in `./graphrag`, so
the system can be used immediately.

### 1. Create a new index with GraphRAG

Install GraphRAG in a virtual environment, initialize a workspace, and put the
source documents in the workspace's `input` directory:

```bash
python -m venv .venv-graphrag
source .venv-graphrag/bin/activate
pip install graphrag

mkdir -p /path/to/graphrag-workspace/input
cp source-texts/*.txt /path/to/graphrag-workspace/input/
graphrag init --root /path/to/graphrag-workspace
```

Configure the provider, API key, chat model, and embedding model in the generated
`.env` and `settings.yaml`, then run:

```bash
graphrag index --root /path/to/graphrag-workspace
```

Indexing invokes model APIs and can be expensive, so start with a small corpus.
See the official [initialization](https://microsoft.github.io/graphrag/config/init/)
and [CLI](https://microsoft.github.io/graphrag/cli/) documentation for the
configuration supported by the installed GraphRAG version.

### 2. Put the Parquet files in this project

Locate the GraphRAG output/artifacts directory that contains the
`create_final_*.parquet` files and copy at least the four required files:

```bash
mkdir -p ./graphrag
cp /path/to/graphrag-output/create_final_documents.parquet ./graphrag/
cp /path/to/graphrag-output/create_final_entities.parquet ./graphrag/
cp /path/to/graphrag-output/create_final_relationships.parquet ./graphrag/
cp /path/to/graphrag-output/create_final_text_units.parquet ./graphrag/
```

Do not put the Parquet files in GraphRAG's `input/` directory; that directory is
only for source documents. To keep the artifacts elsewhere, pass their parent
directory with the global `--graphrag-dir` option.

This revision expects artifact names and schemas from earlier GraphRAG releases.
Recent releases output `documents.parquet`, `entities.parquet`,
`relationships.parquet`, and `text_units.parquet` and changed parts of their
schemas. Renaming those files alone is not sufficient; migrate them to this
schema first:

| Artifact | Required columns |
| --- | --- |
| `create_final_documents.parquet` | `id`, `title`, `raw_content` |
| `create_final_entities.parquet` | `id`, `name`, `type`, `description`, `text_unit_ids` |
| `create_final_relationships.parquet` | `id`, `source`, `target`, `description`, `weight`, `text_unit_ids` |
| `create_final_text_units.parquet` | `id`, `text`, `document_ids`, `entity_ids`, `relationship_ids` |

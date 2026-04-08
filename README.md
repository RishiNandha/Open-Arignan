# Open-Arignan

**Arignar** is the tamil word for the well-read / the knowledgeable / the scholar. **Arignan** is an application that can help scholars, engineers, founders, etc... to maintain a local-first private knowledge base and get queries answered from it.

## Quick-Start

### Option 1: Git Clone + CLI

0. Clone the repo
1. Run the `python setup.py --app-home <install dir>`. This will:
   - Download all the models needed, including the default local answer model and a lighter fallback answer model
   - Create a **bin directory** folder with executables
   - Print the bin directory for your reference
2. Add the bin directory folder to your PATH
3. Try `arignan load "filename.pdf"`
4. Try `arignan ask "relevant question"`

### Option 2: GUI

soon tm ;)

## Key Points

### Behavioral

- **Fully local RAG / Knowledge-base system**: Guilt-free caching of proprietary docs, or things under NDA, or unpublished material.
- **Load and store knowledge over time**: Maybe software help docs as you discover them, research papers as you read them, personal notes, tutorial markdowns, textbooks, etc
- **Session history**: For detailed prompting workflows where the user might choose to ask a series of questions
- **User switches for topic/category**: For advanced users who might wear different "hats" and maintain different knowledge bases for each of them.

### Technical

- **Fully local LLM**: `qwen3:4b-q4_K_M` by default, with `qwen3:0.6b` also provisioned as a lighter answer mode option. Reconfigurable in settings.json
- **4-piece hybrid retreival system**:

  - **map.md files** tell the LLM "where to find what" in the subdirectories
  - **Semantic search** with a vectorDB RAG that fetches context from within the files
  - **User auditable knowledge base** LLM-generated markdowns gives quick context/summary to the LLM.
  - **Keyword search** for retrieving exact keyword matches
- **Incrementally load and deleting of knowledge**:

  - **Loading hook** appends the vector cache to the semantic RAG database, map.md for a quick lookup of "where to find what", and makes the LLM write a summary/knowledge-base markdown
  - **Deleting** allows picking a past load and undoing it gracefully
  - **Optional parameter "hat"** tells the load hook which subdirectory / subdivision of the knowledge base to write to.

### Entry Points

- **CLI Entry Points**:

  - **Session resetting**: Each terminal starts it's own session of maintaining chat history. The user can reset it with this command.
  - **QnA with citations**: Questions can be asked based on the knowledge. Each question is also answered with citations of which directory and file were referred to.
  - **Optional parameter "hat" in QnA** that narrows down where to search thereby improving latency. Set to "auto" by default, which goes through everything
  - **Loading Content**: Takes web url for a blog post / local address to a PDF or Markdown. Takes "hat" parameter which is "auto" by default.
  - **Deleting Content**: First displays the ingestion history with load_IDs. Then user picks the load_IDs to undo the past ingestion.
  - **Saving chat state**: Chat history has to be saved. Otherwise, it get's erase by default
  - **Loading chat state**: To load the saved chat history
- **MCP Entry Points**:

  - **Context Retreival tool**: A client such as copilot or codex or claude code can tap into the knowledge base and retrieve context
  - **Global Map Resource**: The "map" given as a context, so that the MCP client has context of what all knowledge are available
  - (To implement in future) **LLM Setting**: Toggle to use the local LLM in internal functions or start using the MCP Client's LLM

## Detailed Description

### The Hats Concept

Open Arignan organizes knowledge into **namespaces called hats**, representing domains or roles such as Spiking Neural Networks, Entrepreneurship, or Psychology. Each hat has it's own:

- Vector Index
- Keyword (BM25) Index
- Summary knowledge base markdowns
- Original files
- map.md describing what to find where

A global map (global_map.md) provides a high-level view across all hats.

#### Storage Layout

The ingestion log allows for deleting any past loads. An LLM-generated global map describes which hat contains what knowledge.

```
~/.arignan/
├── settings.json
├── ingestion_log.jsonl
└── hats/
    ├── default/
    │   ├── vector_index/
    │   ├── bm25_index/
    │   ├── summaries/
    │   │   └── <topic_folder>/
    │   │       ├── <original_files>
    │   │       └── <markdown_tree>
    │   └── map.md
    ├── <hat_name>/
    │   ├── vector_index/
    │   ├── bm25_index/
    │   ├── summaries/
    │   │   └── <topic_folder>/
    │   │       ├── <original_files>
    │   │       └── <markdown_tree>
    │   └── map.md
    └── global_map.md
```

#### Knowledge-base Organization

The summaries/ directory is LLM-organized and human-auditable. Each subfolder represents a topic grouping and the folder name inferred by the LLM.

Each folder contains the original source file(s) and either a single markdown or a bunch of markdowns. The system gives the LLM the flexibility to do grouping based on size and semantic relatedness:

- **Related documents can be grouped in one folder**: For example: multiple papers on a related coherent topic JEPA can be summarized into one markdown in one folder
- **Large documents can have multiple markdowns**: For example: Behzad Razavi RFIC Design might typically one per section

### Ingestion & Deletion Models

Each document ingestion is a tracked event with its own `load_id`. Accepted inputs:

- Web url to blogs / wikis
- Local path to a PDF
- Local path to a Markdown
- Local path to a folder of PDFs/Markdowns

#### Chunk Parsing

Parsing for the vector index and keyword index are done using headings wherever possible, or with chunk size limits. The rules used are:

- Prefer section-based chunking using detected headings
- Fall back to text splitting for unstructured or long text
- Maintain small overlap between adjacent chunks
- Preserve metadata such as load_id, source path, section / header / page number

#### Embedding

Embedding model used is `BAAI/bge-base-en-v1.5`. This is fixed at build time and can't be changed in settings.json. Each chunk stores:

- The embedding vector
- Canonical chunk text
- Metadata to be used for:
  - Citation: Path, Page Number / Section / Heading
  - Deletion: Load_id

Vector Index is done using Qdrant and HNSW for storing both embedding and metadata. Lexical Index is using BM25

#### Topic Grouping and Segmentation

Grouping of files into a single markdown or segmentation of a single file into multiple markdowns is currently decided by local heuristics, with `max_md_length` assisting the decision. The architecture is still intended to be LLM-guided at this boundary.

**Grouping:**

1. The system runs a retrieval on the database with the given file
2. If there are matches that are close in semantics such as keywords and concepts, the LLM flags a possible merge
3. The merge is then evaluated against how the estimated length of the combined markdown will be (shouldn't be more than `max_md_length`)

**Segmentation:**

1. The system sees the size of the file. If its a book, it goes chapter-wise rightaway (common-case fast)
2. Otherwise, the system estimates the length of a markdown if it had to be write it. And if its more than `max_md_length` then it tries to break it down heading or topic wise.

#### Editting Markdowns and Log

LLM is systematically prompted to update one markdown at a time.

- Knowledge base markdowns: **wiki-styled** and **optimized tightly for token limits.**
- `map.md` to be rich in the following information:
  - Paths to files
  - What to expect from the files, like "RF IC textbook"
  - Any specific keywords, like "Calibre xRC"
- The `global_map.md` to point to the relevant "hat" which would have the relevant map.md. It should have high-level keywords like "JEPA".

**Ingestion Log** is append-only, like commit history and each addition or deletion is logged with the path to reach the relevant changes made it so that the delete function can use this to lookup.

### Deletion

Using the Ingestion log, the files are remove, map.md is updated and the vector and keyword indices are updated. The markdowns is deleted if standalone. If in a grouped setting, then it's regenerated from all the raw sources in the same group again.

### Retrieval Pipeline

1. **Query Expansion**: The system first normalizes the query and adds expansions of abbreviations used
2. **Hat Classification**: When the hat is unspecified, the system first classifies which hat to descend down to
3. **3-way Retrieval**:
   - Qdrant retrieves top-k chunks
   - BM25 retrieves top-k chunks
   - Descending down the maps retrieves the knowledge base markdown. (If the markdown is large, headings are treated as individual chunks).
4. **Reciprocal Rank Fusion**: Chunks that appear in both Qdrant and BM25 are awarded higher score, and the rest are pruned
5. **Cross-Encoder Reranking**: The chunks are reranked. This removes false positives, and removes irrelevant chunks from the markdown. Default cross-encoder used is `BAAI/bge-reranker-v2-m3` (can be changed in settings.json)
6. **Final Answer Mode**: `ask` can use the default local LLM, a lighter local LLM, deterministic retrieval synthesis, or a raw reranked-context dump via `--answer-mode default|light|none|raw`
7. (To implement in future): Adjacent Content Expansion.

### Session Scope

Each time Arignan is called in a new terminal, it starts a new session and **associates the PID of the terminal with the session**.

Each session has:

- A KV Cache in-memory optionally (configured in settings.json)
- A conversation history JSON
- A session ID

KV Cache reset behavior is currently represented in session metadata and is reset either with a timeout, with a soft token limit or upon a session reset.

Active context is maintained in a JSON while the session is active. This can be saved by the user with a command. A user can also load another JSON as the context.

#### Self-Summary Rollover

When the chat history is becoming too long:

- LLM rewrites the dialogue into a session summary
- Older turns are removed from active prompt context
- Session continues with:
  - System prompt
  - Session summary
  - Recent turns
  - Fresh retrieved context
- The session JSON is overwritten with this summarized context (since unlike a chatbot, chat history holds no significance to us)

## Setup

### For Users

1. Setup: `python setup.py --app-home <install dir>`. We will call this the **App home** from now.
2. Add **Bin directory** to PATH. The setup.py will automatically print the bin directory for you.
2. Help: `arignan --help`
3. Load: `arignan load "filename.pdf"`
4. Load with hat: `arignan load "flename.pdf" --hat psychology`
5. QnA: `arignan ask "What is JEPA?"`
6. QnA with hat: `arignan ask "How to use CalibreRC" --hat "IC Design"`
7. Optional answer mode: `arignan ask "What is JEPA?" --answer-mode light`
8. Ingestion Log: `arignan list-loads`
9. Delete a past ingestion: `arignan delete <load_id>`
10. Reset context: `arignan reset-session`
11. Save context: `arignan save-session <path/session_name.json>`
12. Reload context: `arignan load-session <path/session_name.json>`

### For Developers

1. Install dependencies: `python -m pip install -e .[dev]`
2. Run tests: `python -m pytest`
3. Debug Load Command: `arignan load "filepath" --debug`
4. Debug Ask Command: `arignan ask "question" --debug`

## Declaration

Some parts of the repository were generated using LLM-assisted coding applications. There may be potential mismatches between the features described in the README and the implementation. If you come across any, please raise an issue in the github repository!

## Feedback

Please write to me or raise a github issue on any feedback! I would love to hear the pain points while using this, and patch it up!

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

You are free to:

- Share: copy and redistribute the material
- Adapt: remix, transform, and build upon the material

Under the following terms:

- Attribution: You must give appropriate credit
- Non-Commercial: You may not use the material for commercial purposes
- Share-Alike: If you remix or modify, you must distribute under the same license

Full license text: https://creativecommons.org/licenses/by-nc-sa/4.0/

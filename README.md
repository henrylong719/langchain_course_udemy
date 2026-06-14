# LangChain Course Notes and Demos

This repository contains my hands-on Python exercises, demos, and small projects from the Udemy course [Production AI Agents with LangChain + LangGraph](https://www.udemy.com/course/production-ai-agents/).

The code is organized as runnable lesson scripts rather than a single production application. It explores LangChain fundamentals, LCEL chains, prompt templates, output parsing, document loading, embeddings, vector stores, RAG, LangGraph workflows, multi-agent systems, security patterns, LangSmith observability, and testing/evaluation patterns.

## What Is Inside

### LangChain Foundations

| File | Topic |
| --- | --- |
| `main.py` | Environment check and basic OpenAI/Anthropic model calls |
| `core_concepts.py` | LCEL, runnables, streaming, batching, and schema inspection |
| `working_with_llms.py` | Working with different chat model providers |
| `prompt_messages.py` | Basic chat messages and prompt templates |
| `prompt_templates_all.py` | Prompt templates, message placeholders, few-shot prompts, and composition |
| `output_parsers.py` | Structured output basics |
| `output_parsers_final.py` | String, JSON, Pydantic, and structured output parsers |
| `chains.py` | Basic chains, parallel chains, passthroughs, branching, and debugging |
| `smart_bot_section.py` | Smart Q&A bot example with structured responses |

### Document Loading, Embeddings, and Vector Stores

| File | Topic |
| --- | --- |
| `document_loaders.py` | Text, web, lazy, document-structure, and PDF loading |
| `test_splitters.py` | Recursive, markdown, code, and document splitting strategies |
| `embeddings.py` | Embedding model setup examples |
| `embeddings_deep.py` | Embedding basics, similarity, batch embeddings, and caching |
| `vector_stores.py` | Chroma basics, similarity search, metadata filters, persistence, and retrievers |

### RAG and Memory

| File | Topic |
| --- | --- |
| `rag_pipeline.py` | Basic RAG, source-aware RAG, fallback RAG, and structured RAG |
| `advanced_rag.py` | Multi-query retrieval, contextual compression, hybrid search, parent documents, and advanced RAG chains |
| `conversation_memory.py` | Chat memory, multi-session memory, trimming, windowed memory, summary memory, and persistence |
| `research_assistant.py` | AI research assistant project with retrieval, memory, and structured responses |

### LangGraph and Agent Workflows

| File | Topic |
| --- | --- |
| `first_graph.py` | First LangGraph conversation graph |
| `langgraph_core.py` | LangGraph state, reducers, nodes, edges, and graph execution |
| `conditional_edges.py` | Conditional routing, quality loops, and multi-path routing |
| `cycles_loops.py` | Self-correcting and iterative graph loops |
| `human_in_loop.py` | Human approval and iterative review workflows |
| `checkpointing.py` | In-memory and SQLite checkpointing, state inspection, and branching conversations |

### Multi-Agent Systems

| File | Topic |
| --- | --- |
| `tool_calling_agent.py` | Tool-calling agents with LangGraph |
| `supervisor_agent.py` | Supervisor architecture for routing work across specialists |
| `agent_handoffs.py` | Customer-service style handoffs between specialist agents |
| `agent_communication.py` | Message passing, shared state, and blackboard communication patterns |
| `parallel_agents.py` | Parallel agent execution and map-reduce summarization |
| `hierarchical_agents.py` | Hierarchical team-style agent coordination |
| `multi_agent_research_system.py` | Multi-agent research system project with supervisor, parallel search, synthesis, report writing, and quality gates |

### Production Patterns

| File | Topic |
| --- | --- |
| `langsmith_setup.py` | LangSmith tracing and observability setup |
| `security_patterns.py` | Input sanitization, PII detection, LLM guardrails, output validation, and secure pipelines |
| `testing_patterns.py` | Unit tests with mocks, integration tests, LLM-as-judge evaluation, and regression testing |

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- API keys for whichever model providers you plan to run

The project dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT_NAME=langchain-course
```

Only add the keys you need for the script you are running. `.env` is ignored by git.

## Running Examples

Most files are standalone scripts. Run them with `uv`:

```bash
uv run python main.py
uv run python core_concepts.py
uv run python rag_pipeline.py
uv run python first_graph.py
uv run python multi_agent_research_system.py
```

Many scripts have several demo functions and only one may be enabled under the `if __name__ == "__main__":` block. Open the file and switch which demo is called when you want to try a different section.

## Running Tests

Run the pytest-based examples:

```bash
uv run pytest testing_patterns.py
```

Some examples call real LLMs and require API keys. Mock-based tests can run without making provider calls.

## Notes

- This is a learning repository, so scripts intentionally favor clarity and lesson-by-lesson experimentation over shared abstractions.
- Some examples create local Chroma/vector-store state or graph images. Generated artifacts are ignored where appropriate.
- The `docs/` folder contains supporting course/demo materials such as `docs/langchain_demo.pdf`.
- Several demos make paid API calls when executed with real providers.

## Course Reference

These examples follow the themes from [Production AI Agents with LangChain + LangGraph](https://www.udemy.com/course/production-ai-agents/): LangChain foundations, RAG, memory, LangGraph, multi-agent orchestration, security, testing, observability, and production AI-agent patterns.

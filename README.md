# Dynamic Pricing Agent

A LangGraph agent that reprices retail products based on competitor prices
and ratings. The pricing logic itself is a deterministic rule set, not a
model: a competitor price band, a rating-based target, and guardrails on
how far a price can move in a month. Claude, called through Amazon
Bedrock, only comes in for two narrower jobs — sizing a small price move,
and drafting a short brief for some escalations. A price move of more
than 10%, a competitor price move of 30% or more, or a model call that
keeps failing goes to a human instead of being priced automatically.

## Status

Work in progress. What's built and working today:

- The pricing rules themselves, plus the LangGraph pipeline that wires
  them together
- LLM calls with retries and backoff, falling back to human escalation if
  they keep failing
- SQLite staging for decisions, monthly state, and open escalations
- Per-call LLM cost tracking
- Optional Langfuse tracing, off by default until it's configured
- Structured JSON logging
- CI running tests, linting, type checks, a dependency vulnerability
  audit, and CodeQL on every change

Requires Python 3.11 and uv.

## Quick start

```bash
uv sync
cp .env.example .env
uv run pytest
```

The test suite doesn't need AWS credentials or network access — every LLM
call in it is mocked.

## Data

Uses the "Retail Price Optimization" dataset from Kaggle (CC0).

## License

MIT.

# OpenRouter Tool Support Tracker

Automated tracking of function calling (tool) support across different models and providers on OpenRouter.

View the live results at: https://xeophon.github.io/openrouter-tool-check/

## Usage

```bash
uv sync
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

Edit the model list by changing `models.json`.

Run all the models across all providers:
```bash
uv run check_all_models.py
```

Generate the report:
```bash
uv run generate_website.py
```

Sometimes, the model (or the provider) does not properly call tools. Thats why every call is made **three times**.

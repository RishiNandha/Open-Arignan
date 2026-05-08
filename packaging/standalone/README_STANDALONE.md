# Open-Arignan Standalone Package

This package is for users who do not have Python installed.

## First Run

1. Open `Start-Arignan-GUI` from this folder.
2. On first run, Arignan initializes its app-home and downloads the required local models.
3. The GUI opens in your browser.

## Command Line

You can also run:

```text
arignan setup
arignan -gui
arignan load "file.pdf"
arignan ask "question"
```

To avoid Ollama and use the small Hugging Face Qwen model:

```text
arignan setup --llm-backend transformers --llm-model Qwen3-0.6B --llm-light-model Qwen3-0.6B
```

The first setup can take a while because model files are downloaded locally.

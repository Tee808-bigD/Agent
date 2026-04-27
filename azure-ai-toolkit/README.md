# Azure AI Toolkit Starter for SimGuard

This folder gives you a clean starting point for using this project with Azure AI Toolkit in VS Code and for sharing the setup on GitHub.

## What's here

- `agent-config.json`: simple repo-level metadata for the agent
- `prompts/simguard-system.prompt.md`: the main system prompt
- `model-profiles.json`: suggested Azure model profiles
- `evals/sample-tests.jsonl`: starter test cases for quick evaluation
- `.env.example`: environment variable template for Azure OpenAI

## Suggested models

These model choices are based on current Microsoft Learn guidance for Azure OpenAI / Azure AI Foundry models, where availability depends on region:

- `gpt-4.1`: primary chat model
- `gpt-4.1-mini`: lower-cost fast chat model
- `o4-mini`: reasoning-heavy tasks
- `text-embedding-3-large`: retrieval and embeddings

References:

- https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models
- https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new
- https://github.com/microsoft/vscode-ai-toolkit

## How to use in Azure AI Toolkit

1. Open this repository in VS Code.
2. Install the Azure AI Toolkit extension.
3. In AI Toolkit, create or open an Agent Builder project.
4. Paste the contents of `prompts/simguard-system.prompt.md` into the system prompt.
5. Use one of the model profiles in `model-profiles.json` when creating your Azure model deployment.
6. Add the values from `.env.example` to your local environment or project secrets.
7. Use `evals/sample-tests.jsonl` as starter test cases in Agent Builder or your own evaluation flow.

## GitHub-friendly setup

Commit this folder as part of your repo so the prompt, model choices, and test cases live with the code.

Recommended next files to add later:

- `.github/workflows/` CI for linting and tests
- project-specific MCP server config
- deployment notes for Azure AI Foundry or Azure Container Apps

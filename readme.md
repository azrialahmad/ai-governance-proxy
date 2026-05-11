# Gatekeeper AI Proxy

Gatekeeper is a secure, intelligent AI proxy system designed to sit between users and downstream Large Language Models (LLMs). It provides a robust pipeline for sanitizing inputs, dynamically routing requests based on complexity, and rigorously evaluating outputs to prevent hallucinations.

## Features

- **Intelligent Prompt Routing**: Analyzes prompt length and complexity. Routes simple prompts to faster models (like Gemini 1.5 Flash) and complex/code-related prompts to more capable models (like Llama-3-70b via Groq).
- **PII Scrubbing**: Uses local models (e.g., Ollama `llama3.2`) with few-shot prompting to accurately detect and redact Personal Identifiable Information (names, emails, phone numbers) before prompts are sent to external APIs, without over-scrubbing common nouns.
- **Hallucination Judge (Evaluation-as-a-Service)**: A standalone evaluation engine that acts as an LLM-as-a-Judge. It compares AI responses against a curated "Golden Set" of facts. Responses scoring below 70% are automatically blocked to prevent the propagation of hallucinations.

## Current Project Status

We have completed **Phase 1** (Data Fixtures) and **Phase 2** (Hallucination Judge). 

**We are currently entering Phase 3.**

### Phase Roadmap

- [x] **Phase 1: Data Fixtures** 
  - Created a Golden Set of Q&A pairs for fact-checking.
  - Assembled diverse adversarial jailbreak prompts for testing.
- [x] **Phase 2: Hallucination Judge**
  - Integrated local LLM-as-a-Judge to evaluate factual accuracy.
  - Implemented strict blocking mechanisms for responses scoring < 70%.
- [ ] **Phase 3: The Adversarial Shield** *(Current Phase)*
  - Building a Red-Team runner to continuously test the proxy against jailbreaks and adversarial attacks.
- [ ] **Phase 4: The Metrics Module**
  - Cost vs. Quality analysis for the routing decisions.

## Setup & Execution
The proxy runs on FastAPI and relies on local Ollama models for security checks, alongside Groq and Gemini for standard completions.

*(More detailed setup instructions to be added as the project evolves)*

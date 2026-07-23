*This project has been created as part of the 42 curriculum by aidbrm.*

# Call Me Maybe - Introduction to Function Calling in LLMs

## Description
**Call Me Maybe** is a Python-based project designed to translate natural language prompts into structured, machine-executable function calls using a small language model (`Qwen/Qwen3-0.6B`).

Small language models often struggle to consistently output valid structured data (such as JSON). This project addresses that challenge by implementing **constrained decoding**, modifying logit probabilities during token generation to guarantee 100% syntactically and semantically valid JSON outputs matching expected function schemas.

---

## Why Constrained Decoding?

Traditional prompting relies on the model "guessing" the right format, which works most of the time with large models but breaks down quickly with smaller ones. Instead of hoping the model produces valid JSON, this project intervenes directly in the generation process:

- At each decoding step, the model's logits are masked so that only tokens consistent with the target grammar/schema remain viable.
- This removes the possibility of malformed keys, missing brackets, wrong types, or hallucinated fields.
- The result is a small, fast model that behaves as reliably as a much larger one when it comes to structured output — without any fine-tuning.

---

## Features

- Natural language → structured function call translation
- Deterministic, schema-valid JSON output via constrained decoding
- Works with lightweight models (`Qwen/Qwen3-0.6B`), keeping inference fast and resource-friendly
- Easily extensible to new function schemas

---

## Instructions

### Prerequisites
* **Python 3.10+**
* Package manager: `uv` (recommended)

### Installation
Install project dependencies and sync the environment:
```bash
make install
```

### Usage
Once dependencies are installed, run the project with:
```bash
make run
```

You can then provide natural language prompts, and the model will output a structured function call matching the expected schema.

---

## Project Structure
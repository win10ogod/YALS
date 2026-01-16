# Agent Notes: YALS

## Summary
- YALS is an OpenAI-compatible API server built with Deno/TypeScript.
- Inference runs through llama.cpp via a C++ shared library exposed to Deno FFI.
- API surface includes OpenAI-style completions/chat and an Anthropic-style messages endpoint.

## Tech Stack
- Deno 2.x, TypeScript, Hono, Zod
- llama.cpp (embedded as a submodule-like folder and built via CMake)
- Jinja chat templates (HuggingFace-style)

## Entry Points
- `main.ts`: loads config, bindings, model, then starts the API server.
- `api/server.ts`: Hono app + routers + middleware.
- `bindings/bindings.ts`: FFI model loading, tokenizer, and generation.
- `bindings/server/*.cpp|hpp`: C++ inference engine and FFI glue.

## Key Directories
- `api/`: routers, request/response schemas, and generation helpers.
- `common/`: config parsing, logging, templating, sampling overrides.
- `bindings/`: Deno FFI bindings + C++ shared library sources.
- `llama.cpp/`: vendored llama.cpp source (used by CMake).
- `templates/`: Jinja prompt templates (including tool-use variants).
- `sampler_overrides/`: preset sampler overrides.

## Configuration
- Copy `config_sample.yml` to `config.yml`.
- Important keys (under `model`):
  - `model_name`, `model_dir`
  - `max_seq_len`, `cache_size`
  - `prompt_template` and `chat_template_kwargs`
  - `mmproj`, `mmproj_use_gpu`, `image_min_tokens`, `image_max_tokens`
- Sampler overrides are set under `sampling.override_preset`.

## Build and Run
- Dev: `deno task dev`
- Run: `deno task start`
- Build binary: `deno task build` or `deno task build-win`
- Rebuild bindings:
  - Unix: `deno task bindings`
  - Windows: `deno task bindings-win`

## API Surface (high-level)
- Health: `GET /health`
- Models: `GET /v1/models`, `GET /v1/model`
- Model lifecycle: `POST /v1/model/load`, `POST /v1/model/unload`
- Templates: `GET /v1/templates`, `POST /v1/template/switch`
- Tokenization: `POST /v1/token/encode`, `POST /v1/token/decode`
- OpenAI API: `POST /v1/completions`, `POST /v1/chat/completions`
- Anthropic API: `POST /v1/messages`
- Auth: `GET /v1/auth/permission`

## Notes and Pitfalls
- For multimodal, `mmproj` must match the model and llama.cpp must support the projector type.
- Tool-use templates use `<prompt_template>_tool_use.jinja` when present.
- Context shift/self-extend is disabled for multimodal paths.

# Image sizes: stages and `.dockerignore`

How the `api` image's size and build depend on its build stages and on what the build context lets in (Phase 10). Measured on 2026-10-15 with Docker 29.8.1 (BuildKit), WSL2 on an x86-64 machine, from the same commit, with `scratchpad` variants of `docker/api/Dockerfile`. Image size is `docker image inspect --format '{{.Size}}'`.

| Variant | Stages | Build context | Bytecode compiled | Context sent | Build time | Image |
| --- | --- | --- | --- | --- | --- | --- |
| A | single (`COPY . .`) | everything | no | 7.05 GB | 1,478 s | 11 GB |
| B | single (`COPY . .`) | allowlist | no | 18 MB | 30 s | 639 MB |
| E | multi | allowlist | no | 18 MB | 13 s | 551 MB |
| C (shipped) | multi | allowlist | yes | 18 MB | 17 s | 724 MB |

**The ignore file is what matters.** Without one, `COPY . .` takes in the 6.4 GB development `.venv` (with CUDA torch), `.git`, the DVC cache and every report: the context is 7.05 GB and the image 11 GB. The allowlist (`docker/api/Dockerfile.dockerignore`, everything excluded but what the Dockerfile copies) brings the context to 18 MB (`pyproject.toml`, `uv.lock`, `src/`, `params.yaml`, the 16 MB index and the 0.9 MB exported retriever).

**Multi-stage saves less than expected here.** Compared like for like (E against B, neither compiling bytecode), it saves 88 MB: the uv binary (56 MB) and the build context stay in the builder. It saves no compilers, because every dependency installs from a wheel. B looked smaller than the shipped image only because uv hard-linked its 282 MB virtual environment to its own cache, so the cache left in the image cost almost nothing.

**Precompiled bytecode costs 173 MB** (C against E): `UV_COMPILE_BYTECODE=1` writes every module's `.pyc` at build time, so a container doesn't compile them on each start. The startup time it saves isn't measured here.

**Notes on reading the table.**
- BuildKit sends only what changed since the last build, so the "context sent" figure is reliable only for a first build. The 18 MB rows give the allowlist's size on disk (`du -sc` of its paths), not the transfer.
- A multi-stage build with no ignore file produced the same 724 MB image as C, since it copies named paths only, but sent the full context on a cold builder.

**The other images.** `models` is 2.3 GB on CPU torch, with its weights in a volume, not in the image. `ollama/ollama:0.40.4` is 9.4 GB, most of it the CUDA libraries it ships for GPUs.

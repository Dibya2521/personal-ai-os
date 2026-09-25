# 0006. llama.cpp's server and Qwen3.5-4B for the local model

Status: accepted. The order in which builds are tried is reasoned, not yet
measured; a measurement that changes it gets a new record.

## Context

The local model answers when the remote one should not or cannot: background
jobs, an open circuit, a spent budget, no network. It has to run on an
ordinary laptop with no CUDA GPU and limited disk, and equally on a machine
with an NVIDIA GPU or an Apple Silicon Mac. It should see images, since the
remote models can, and it must not become a door into the machine for other
programs.

## Decision

- **Runtime: llama.cpp's own `llama-server` binary**, started and supervised
  as a service. It speaks the OpenAI-compatible chat API, so the gateway's one
  adapter serves it unchanged, and it enforces a JSON Schema reply with a
  grammar.
- **Model: Qwen3.5-4B in the Q4_K_M quantisation** with its F16 vision
  projector, from the `unsloth/Qwen3.5-4B-GGUF` repository at a pinned
  revision (`e87f176479d0855a907a41277aca2f8ee7a09523`). Neither Qwen nor
  ggml-org publishes GGUF files of Qwen3.5 (their Hugging Face APIs return not
  found for it, 2026-09-23), so that repository is the source. Download:
  2.55 GB for the model and 0.63 GB for the projector (binary gigabytes, as
  `synthia models` shows them).
- **Every file is pinned by SHA-256**, and llama.cpp is pinned to one build,
  `b11130`, with the digests GitHub publishes for its assets. llama.cpp
  publishes several builds a day, all marked prerelease, and its "latest"
  release carries none of the binaries, so a build tag is the only stable pin.
  A download gets its final name only after its size and digest match, so a
  file under that name has always been verified; one that does not match is
  deleted.
- **Builds are tried fastest first: Metal, CUDA, Vulkan, then CPU.** The
  default install takes the likely fastest build for the machine (CUDA with an
  NVIDIA driver, otherwise Vulkan, which covers AMD and Intel GPUs) plus the
  CPU build as the floor. A build that fails to start or to answer its health
  check within 180 seconds is skipped and the next one launched. The layers
  offloaded to a GPU are left to llama.cpp's default, which fits as many as its
  memory holds, so a small GPU still helps. With nothing installed, SYNTHIA
  runs on the remote model alone, and `synthia doctor` says so.
- **CUDA builds use CUDA 12** (12.4 on Windows, 12.8 on Linux), with the
  runtime archive they need. The smaller CUDA 13 builds are also published;
  neither is tested yet, since no machine here has an NVIDIA GPU.
- **The server is private to SYNTHIA**: it listens on 127.0.0.1 only, on a
  port chosen per launch, with a random 32-byte key made per launch and passed
  in its environment rather than on the command line, where other users could
  read it. Its web interface is switched off (`--no-webui`) and so is its own
  network access (`--offline`).

## Options considered

- **Ollama.** Easy to install, but it runs llama.cpp underneath, so it adds a
  second daemon and its own model store without adding speed.
- **llama-cpp-python.** The same engine as a native extension inside
  SYNTHIA's process, so a crash in the model takes the whole assistant down,
  and a build for each accelerator has to be compiled or found as a wheel. A
  separate server fails alone and restarts.
- **Gemma-4-E4B** (4.98 GB plus a 0.99 GB projector, as Hugging Face lists
  them). It also hears audio, but its model card scores are lower than
  Qwen3.5-4B's on the card figures compared: MMLU-Pro 69.4 against 79.1, GPQA
  58.6 against 76.2, MMMU-Pro 52.6 against 66.3. Speech will be handled by a
  dedicated streaming recogniser, so audio input in the language model is not
  needed.
- **Qwen3.5-9B** (5.68 GB plus 0.92 GB). Stronger, but more than twice the
  download and memory for a model that serves as the fallback. It stays one
  setting away (`SYNTHIA_LOCAL_MODEL`) for a machine that can carry it.
- **Qwen3-4B.** Text only; it cannot take the images the remote model can.

## Consequences

The install is about 3.2 GB on a machine without an NVIDIA GPU, checked
against the disk budget before anything is downloaded. Changing the model or
the llama.cpp build is a catalogue entry with new digests. Three values are
still reasoned rather than measured, and will be measured with the model
running: whether Vulkan is faster than the CPU on an integrated GPU, the
180-second start timeout, and the memory a 32,768-token context costs.
Whether Qwen3.5-4B calls tools reliably through llama.cpp's default chat
template is also unverified until then; the router already assumes it can.

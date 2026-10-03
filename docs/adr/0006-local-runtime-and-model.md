# 0006. llama.cpp's server and Qwen3.5-4B for the local model

Status: accepted. The order in which builds are tried was reasoned first and
then measured on one class of machine (see Measured); a measurement that
changes it gets a new record.

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
the llama.cpp build is a catalogue entry with new digests. Tool calling was
measured later: Qwen3.5-4B made 60 of 60 calls correctly through llama.cpp's
default chat template (decision record 0010).

## Measured

`benchmarks/local_inference.py` starts the server the way `synthia chat` does
and sends the same 375-token prompt five times after one warm-up request, each
opening with its own number so the server cannot reuse a cached prompt (its log
confirms all 375 tokens were processed every time), asking for at most 128
tokens with thinking off.

On a laptop with an integrated GPU and no NVIDIA GPU, at 2 threads while other
work kept the processor about half busy, with the 32,768-token context per
slot (4 slots):

| Build | Load | First token | Prompt tokens/s | Answer tokens/s | Server memory |
|---|---|---|---|---|---|
| Vulkan | 14.9 s | 7.9 to 10.1 s (median 9.3) | 37 to 47 (median 40) | 4.4 to 5.1 (median 4.6) | 6,183 MB peak |
| CPU | 12.5 s | 41.9 to 82.0 s (median 59.7) | 4.6 to 8.9 (median 6.3) | 2.0 to 3.3 (median 2.6) | 6,600 MB peak |

Vulkan reaches the first token about six times sooner and answers about 1.8
times faster, so trying it before the CPU build holds on this class of machine.
Two threads limit the CPU build more than the Vulkan one, so the gap at the
default thread count is expected to be smaller (not yet measured; that run
waits for an idle machine). Loading took 12.5 to 15 seconds, well inside the
180-second start timeout, minutes after earlier runs had read the model; a
first load after a restart was not measured here.

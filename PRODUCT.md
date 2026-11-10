# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Reefer truck fleet operators, logistics dispatchers, cold-chain operations controllers, and transport drivers on Indian national highways who communicate over noisy cellular telephony under intense cabin noise.

## Product Purpose

Vaani (वाणी) is an industrial voice dispatcher that prevents catastrophic cold-chain cargo loss and vehicle misdispatch by guaranteeing that spoken driver reports are confirmed aloud via text-to-speech before any fleet action or ticket is executed.

## Positioning

Unlike conventional conversational AI chatbots or black-box voice assistants that directly trigger backend actions from noisy speech-to-text extractions, Vaani guarantees zero silent misdispatches through a deterministic finite state machine (FSM) safety gate and spoken read-back verification protocol.

## Operating Context

- 8 kHz narrowband GSM / cellular telephony channels with low signal-to-noise ratios (SNR 10 dB and 5 dB) dominated by highway truck cabin roar, horns, and engine rumble.
- Hinglish code-switching ("reefer ka temperature 18 degree ho gaya toll plaza se 42 km aage").
- High-stakes cold-chain logistics where mishearing 18°C as 80°C or -14°C as +14°C spoils perishable pharmaceutical or ice-cream cargo worth lakhs.
- Desktop and control-tower operations consoles used by logistics supervisors monitoring active dispatches and fleet telemetry.

## Capabilities and Constraints

- Audio channel degradation simulator (Clean 16 kHz, 8 kHz GSM Bandpass, 10 dB SNR Highway Truck, 5 dB SNR Heavy Roar).
- Speech-to-Text integration supporting Fine-Tuned Whisper-LoRA (local GPU), Gnani Prisma STT (cloud), and offline acoustic confusion models.
- Structured Pydantic extraction gate validating physical bounds (-40°C to +50°C for reefer cargo).
- Finite State Machine dialogue controller enforcing LISTENING -> EXTRACTING -> CONFIRMING -> (CORRECTING) -> ACTING -> DONE / ESCALATING.
- Gnani Timbre spoken audio read-back aloud to driver.
- SQLite persistent fleet action log and telemetry tracking.
- Interactive empirical benchmark evaluator on 100 test clips across 4 noise conditions.
- Offline acoustic confusion audit queue and chunked dataset training ledger with auto-purge.

## Brand Commitments

- Brand Name: Vaani // Voice AI Dispatcher (वाणी)
- Competition: Great Indian AI Internship Challenge by Gnani AI (Track: Noisy Telephonic Audio)
- Core Ethos: "The language model never decides whether a dispatch happens. The code gate checks validity, reads back aloud, and only explicit confirmation unlocks action."

## Evidence on Hand

- 100 benchmark audio clips spanning 25 phonetic trap utterances across 4 channel conditions in `benchmark/audio/`.
- Summary metrics and raw CSV results showing 100% error interception in `benchmark/results/`.
- 66 passing automated pytest regression tests in `tests/`.
- Working SQLite database with vehicle fleet records and action audit log in `data/fleet.sqlite`.

## Product Principles

1. **Safety Gate Invariant:** Language models propose; deterministic code validates; human confirmation authorizes. No autonomous action without spoken verification.
2. **Operate Clarity Over Novelty:** In high-stress fleet operations, scanability, state clarity, and unambiguous data readouts outrank decorative flair.
3. **Respect Physical Realities:** Telephony bandwidth, acoustic noise, and physical temperature bounds are immutable facts that the UI must reflect transparently.
4. **Closed-Loop Traceability:** Every voice report, extraction, correction round, and dispatch ticket is auditable end-to-end.

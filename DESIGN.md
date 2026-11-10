# Design System: Vaani Telematics Command Console

<!-- impeccable:design-schema 1 -->

## Visual World

Industrial Cold-Chain Telematics Control Tower. Precision slate-graphite surfaces, high-contrast operational status signaling, calibrated typography, and an unambiguous safety gate interlock.

## Platform & Mode

- Platform: Web (Streamlit Application)
- Mode: **Operate** (High-density telemetry, scanability, real-time audio analysis, safety gate interlock, zero decorative slop)

## Color Architecture (Restrained Operational System)

- **Base Canvas:** `#0D1117` (Deep Obsidian Slate)
- **Elevated Surfaces:** `#161B22` (Substrate Panel)
- **Interactive & Raised:** `#21262D` (Card / Section Surface)
- **Hairline Dividers:** `#30363D` (1px precision border)
- **Primary Text:** `#F0F6FC` (High legibility crisp contrast, 13.8:1 on base)
- **Secondary / Metadata:** `#8B949E` (Neutral slate, 5.2:1 contrast)
- **Status Signals (Semantic Roles Only):**
  - **Verified / Safe:** `#238636` / `#2EA043` (Pass, Dispatched, Confirmed)
  - **Telemetry / ASR:** `#1F6FEB` / `#58A6FF` (Live Voice Stream, Active Processing)
  - **Acoustic Warning:** `#D29922` / `#F0883E` (Noise Degradation, Correction Required)
  - **Breach / Escalation:** `#DA3633` / `#F85149` (Critical Setpoint Excursion, Escalated to Human)

## Typography

- **Primary UI & Headings:** `Archivo`, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif
  - Weights: 400 (Body), 500 (Labels/Controls), 600 (Subheadings), 700 (Titles)
  - Fixed rem scale with tight ratio (1.15x) for data-dense telemetry
- **Data, Telemetry & Code:** `JetBrains Mono`, monospace
  - `font-variant-numeric: tabular-nums` for all temperatures, distances, latencies, timestamps, and percentages

## Craft Floor Guardrails Enforced

- **Zero Gradient Text:** Authority comes from font-weight and crisp contrast.
- **Zero Heavy Side-Tab Borders:** No `border-left: 4px` strips; subtle 1px semantic borders and background tints only.
- **Zero Nested Lazy Cards:** Clean structural layout using grid dividing rules.
- **Engineered SVG Iconography:** Uniform 1.5px/2px stroke technical icons instead of random emojis.
- **Custom Browser Surfaces:** Custom scrollbars, focus rings (`#58A6FF`), selection highlight (`#1F6FEB`).

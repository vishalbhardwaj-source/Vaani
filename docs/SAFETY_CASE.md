# Vaani Safety Case & Adversarial Telematics Gate

## 1. Safety Principles & Industrial Requirements

Logistics voice assistants operate in high-risk environments. A misplaced decimal point, dropped negative sign, or acoustic confusion between *18°C* and *80°C* can destroy sensitive pharmaceuticals or spoil ice cream cargo.

Vaani enforces 5 non-negotiable safety guardrails:

1. **Bi-Directional Sign-Flip & Dropped-Minus Protection:** Sub-zero targets (e.g. −22°C ice cream) reported as positive (+14°C) trigger an immediate `SIGN_FLIP_DETECTED` challenge.
2. **Author-Configured Cargo Tolerance Bands:** Informed by WHO PQS (Vaccines: ±4°C), FSSAI (Chilled Dairy: ±8°C), Codex (Deep Freeze: ±12°C), and APEDA (Produce: ±15°C). Spoken readings exceeding tolerance trigger mandatory number repeat verification.
3. **Decoupled Risk Monitoring:** The ASR gate answers *"Was this transcribed accurately?"*, while the excursion monitor answers *"Is cargo degrading?"*. True failures matching sensor are permitted without nuisance challenges, immediately triggering critical dispatch alarms.
4. **Anti-Echo Loop Defense:** Rejects acoustic reflections of agent read-backs from cab speakers.
5. **Caller ID Verification & Retry Cap:** Strictly caps unverified vehicle retries at 2 before supervisor transfer.

---

## 2. 16 Adversarial & Boundary Scenario Benchmark

Evaluated across 16 adversarial scenarios in [`benchmark/results/adversarial_gate_results.csv`](file:///c:/Users/visha/Downloads/VoiceAI/benchmark/results/adversarial_gate_results.csv):

| Scenario ID | Vehicle ID | Cargo Class | Target (°C) | Reported (°C) | Divergence | Gate Action | Reflexive Confirms Blocked | Verified Repeat Permitted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `adv_01` | HR-55-TC-1102 | Ice Cream | -22.0 | +12.0 | 34.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_02` | HR-55-TC-1102 | Ice Cream | -22.0 | +18.0 | 40.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_03` | HR-55-TC-1102 | Ice Cream | -22.0 | +25.0 | 47.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_04` | HR-55-TC-1102 | Ice Cream | -22.0 | -8.0 | 14.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_05` | HR-55-TC-1102 | Ice Cream | -22.0 | -10.0 | 12.0°C | ALLOW | N/A (Boundary) | Allowed |
| `adv_06` | HR-55-TC-1102 | Ice Cream | -22.0 | -12.0 | 10.0°C | ALLOW | N/A (Safe) | Allowed |
| `adv_07` | MH-12-RN-8819 | Vaccine Cold-Chain | -20.0 | -14.0 | 6.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_08` | MH-12-RN-8819 | Vaccine Cold-Chain | -20.0 | -18.0 | 2.0°C | ALLOW | N/A (Pharma Safe) | Allowed |
| `adv_09` | DL-01-AA-4092 | Chilled Dairy | +4.0 | -24.0 | 28.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_10` | DL-01-AA-4092 | Chilled Dairy | +4.0 | +14.0 | 10.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_11` | DL-01-AA-4092 | Chilled Dairy | +4.0 | +11.0 | 7.0°C | ALLOW | N/A (Dairy Safe) | Allowed |
| `adv_12` | DL-01-AA-4092 | Chilled Dairy | +4.0 | +40.0 | 36.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_13` | GJ-01-BX-7201 | Fresh Produce | +6.0 | -22.0 | 28.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_14` | GJ-01-BX-7201 | Fresh Produce | +6.0 | +24.0 | 18.0°C | CHALLENGE | 100% Blocked | Allowed |
| `adv_15` | GJ-01-BX-7201 | Fresh Produce | +6.0 | +19.0 | 13.0°C | ALLOW | N/A (Produce Safe) | Allowed |
| `adv_16` | DL-01-AA-4092 | Chilled Dairy | +4.0 | +40.0 | 36.0°C | CHALLENGE | Allowed with Audit Flag | Allowed |

**Residual Risk Accounting (`adv_16`):** If a driver explicitly confirms an erroneous reading upon repetition (e.g. *"haan 40 degree confirm"*), the gate allows dispatch but sets `residual_risk_flag = True`, notifying human control tower supervisors immediately.

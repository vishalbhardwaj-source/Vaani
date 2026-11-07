# 60-Second Video Demonstration Guide: The 18°C vs 80°C Acoustic Catch

**Target Category:** *60-Second Demo Video Award*  
**Submission Tags:** `@GnaniAI`  
**Required Hashtags:** `#GnaniAI #VoiceAI #GreatIndianAIChallenge`

---

## 1. 60-Second Video Script & Timing Breakdown

| Time | Scene / Visual | Voiceover / Spoken Audio (Hinglish) | On-Screen Callout |
| :---: | :--- | :--- | :--- |
| **0:00 - 0:08** | Title Card + Truck Cabin Noise Video | *"On noisy Indian highways at 70 km/h, speech recognition frequently confuses 18°C with 80°C. In cold-chain logistics, that single error spoils an entire ₹25 Lakh consignment."* | **Problem: Fatal Acoustic Confusion (18°C ↔ 80°C)** |
| **0:08 - 0:20** | Terminal / Dashboard with real audio clip playing (`utt_01_telephony_10db.wav`) | Driver voice speaks: *"Bhaiya reefer temperature 18 degree ho gaya hai toll plaza ke paas"*<br>Gnani Prisma STT displays: `80 degree` (Acoustic Trap Triggered!) | **Gnani Prisma STT transcribes noisy phone audio** |
| **0:20 - 0:34** | Vaani State Machine & Timbre Read-Back | Gnani Timbre speaks readback: *"Temperature 80 degree, Panipat toll pe. Sahi hai na bhaiya?"*<br>Driver immediately cuts in: *"Nahi, 18 degree confirm!"* | **Gnani Timbre TTS conversational read-back** |
| **0:34 - 0:48** | Gate Interception & Auto-Correction | Agent intercepts divergence: *"Theek hai, correct kar liya: 18 degree verify ho gaya."*<br>Telematics check corroborates reefer temperature. Ticket dispatched! | **Deterministic Gate intercepts error; Zero wrong dispatches** |
| **0:48 - 1:00** | Architecture Summary Card | *"Vaani confirms before it acts. Built with Gnani Prisma, Evon, and Timbre for the Great Indian AI Challenge."* | **Stack: Prisma + Evon + Timbre + Python State Machine** |

---

## 2. Step-by-Step Recording Instructions

1. **Launch the Demo Scenario:**
   ```bash
   python demo.py
   ```
   (Scene 3 specifically executes the 18°C vs 80°C trap with Prisma STT, Evon NLU, and Timbre TTS).
2. **Launch the Telematics Dashboard:**
   ```bash
   streamlit run app/dashboard.py
   ```
   Navigate to the **Live Dispatch Console** tab to show the real-time ticket stream and telematics corroboration.
3. **Capture Screen with OBS or Windows Game Bar (`Win + Alt + R`):**
   - Record terminal running `python demo.py` alongside the Streamlit dispatch map.
   - Play audio output from speakers to capture genuine Timbre synthesis.
4. **Post to LinkedIn / X:**
   ```markdown
   Excited to submit Vaani for the Great Indian AI Challenge by @GnaniAI! 🚛🎙️

   On noisy highways, ASR easily mishears 18°C as 80°C, risking severe cold-chain spoilage. 
   Vaani uses Gnani Prisma, Evon, and Timbre combined with a deterministic safety gate: 
   the model proposes details, code verifies against telematics, and only affirmative driver confirmation unlocks dispatch.

   Watch how our read-back catches an 18°C vs 80°C acoustic error in real-time!

   #GnaniAI #VoiceAI #GreatIndianAIChallenge #BuildWithAI #LogisticsTech
   ```

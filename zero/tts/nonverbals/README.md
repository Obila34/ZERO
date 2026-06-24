# Non-verbal clips (Piper expressive path)

When the TTS engine is **Piper**, the orchestrator splices short pre-recorded
sounds in wherever the LLM emits a cue tag. Drop mono WAV files here named:

| Cue tag       | File          |
|---------------|---------------|
| `[laughs]`    | `laugh.wav`   |
| `[chuckles]`  | `chuckle.wav` |
| `[sighs]`     | `sigh.wav`    |
| `[hmm]`       | `hmm.wav`     |
| `[pause]`     | *(none — rendered as silence)* |

Tips:
- Keep them short (0.3–1.5 s) and recorded in the **same voice character** as the
  Piper voice so the splice isn't jarring.
- Any sample rate is fine — the orchestrator resamples to the Piper voice rate.
- Missing files degrade gracefully to a short pause (logged as a warning).

These are unused when `tts.engine: fish` — Fish performs `(laughing)` etc. itself.

"""ZERO — offline conversational voice robot for the Raspberry Pi 5.

First step toward a conversational humanoid: talk to it in English, it replies
in a natural human-sounding voice. Every pipeline stage (wake / vad / stt / llm
/ tts) lives behind a thin interface so engines can be swapped via config.yaml.
"""

__version__ = "0.1.0"

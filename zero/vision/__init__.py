"""Vision subsystem — ZERO's eyes.

The camera + local detection/color loop runs continuously in the background
(``Eyes``), exactly like a person whose eyes are always open. The wake word does
not *start* vision; it makes ZERO *pay attention* to what it has already been
seeing. On a conversational turn the orchestrator snapshots the current scene
(instant, already computed) and — for visual questions — asks the GPU vision
server for grounded distance/bearing facts and optionally hands the keyframe to a
multimodal LLM. One brain, one memory, one voice.
"""

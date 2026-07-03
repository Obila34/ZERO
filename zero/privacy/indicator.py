"""ListeningIndicator — the visible "I can hear you" signal.

On the current body (Pi in a shell, no screen) the honest option is an LED on
a GPIO pin: solid while ZERO is in a conversation (mic hot + transcribing),
off when idle (wake-word matching only, nothing leaves the device). Without a
configured pin it degrades to state-transition log lines, so the state is
still observable over `journalctl`.
"""
from __future__ import annotations

from zero.utils.logging import get_logger

log = get_logger("privacy.indicator")


class ListeningIndicator:
    def __init__(self, gpio_pin: int | None = None):
        self._led = None
        self._state = "idle"
        if gpio_pin is not None:
            try:
                from gpiozero import LED  # present on Pi OS; optional elsewhere

                self._led = LED(int(gpio_pin))
                log.info("listening LED on GPIO %d", gpio_pin)
            except Exception as e:
                log.warning("no LED on GPIO %s (%s) — using log indicator only",
                            gpio_pin, e)

    def set_state(self, state: str) -> None:
        """idle | listening | thinking | speaking. Conversation states light
        the LED; idle turns it off."""
        if state == self._state:
            return
        self._state = state
        hot = state in ("listening", "thinking", "speaking")
        if self._led is not None:
            try:
                (self._led.on if hot else self._led.off)()
            except Exception as e:
                log.debug("LED write failed: %s", e)
        log.debug("indicator: %s (%s)", state, "on" if hot else "off")

    def close(self) -> None:
        if self._led is not None:
            try:
                self._led.off()
                self._led.close()
            except Exception:
                pass

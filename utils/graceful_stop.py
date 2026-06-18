"""Graceful stop mechanism for training loops.

Provides a cross-platform daemon thread that monitors keyboard input and
sets a threading.Event when a configurable key (default: ``'q'``) is pressed.
The main training loop should periodically call :meth:`stop_requested` and break
cleanly when it returns ``True`` — allowing the best model checkpoint to be
saved before exit.

Key principles:
    - **Non‑blocking**: the listener runs in a background daemon thread.
    - **Cross‑platform**: uses ``msvcrt`` on Windows and ``select`` / ``tty``
      on Unix/Linux/macOS.
    - **Clean teardown**: restores terminal attributes on :meth:`stop` and via
      :func:`atexit.register`.
    - **Self‑contained**: no external dependencies beyond the Python standard
      library.

Usage:
    >>> from utils.graceful_stop import GracefulStop
    >>> stopper = GracefulStop(stop_key='q')
    >>> stopper.start()
    >>> for epoch in range(max_epochs):
    ...     if stopper.stop_requested():
    ...         logger.info("User requested stop — terminating gracefully")
    ...         break
    ...     # ... training code ...
    >>> stopper.stop()
"""

import atexit
import logging
import platform
import queue
import sys
import threading
import time

logger = logging.getLogger(__name__)


class GracefulStop:
    """A key‑listener that signals training loops to stop gracefully.

    Launches a daemon thread that reads keyboard input. When the configured
    *stop_key* is pressed, a :class:`threading.Event` is set. The event can
    be polled from the main thread via :meth:`stop_requested`.

    Attributes:
        event (threading.Event): Set when the stop key has been pressed.
        stop_key (str): The single character (or escape sequence) that
            triggers the stop. Default is ``'q'``.
    """

    def __init__(self, stop_key: str = "q") -> None:
        """Initialise the graceful-stop listener.

        Args:
            stop_key: The key character that triggers termination. Use
                ``'\\x1b'`` for the Escape key, ``'q'`` for the letter Q, etc.
                Defaults to ``'q'``.
        """
        self.event: threading.Event = threading.Event()
        self.stop_key: str = stop_key
        self._listener_thread: threading.Thread | None = None
        self._running: bool = False
        # Track whether we changed terminal settings (Unix)
        self._old_term_settings: list | None = None
        # Register cleanup on interpreter exit
        atexit.register(self._cleanup_terminal)

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background key‑listener daemon thread.

        The thread reads from ``sys.stdin`` and sets :attr:`event` when
        *stop_key* is detected. On Unix the terminal is temporarily switched
        to raw mode so that single keystrokes are available immediately.
        """
        if self._running:
            logger.debug("GracefulStop listener is already running.")
            return

        self._running = True
        self.event.clear()

        self._listener_thread = threading.Thread(
            target=self._listen,
            name="graceful-stop-listener",
            daemon=True,
        )
        self._listener_thread.start()
        logger.info(
            "Graceful stop listener started — press '%s' to terminate "
            "training gracefully at the next epoch boundary.",
            self.stop_key,
        )

    def stop(self) -> None:
        """Stop the key‑listener thread and restore terminal settings."""
        self._running = False
        self._cleanup_terminal()

    def stop_requested(self) -> bool:
        """Return ``True`` if the user has pressed the stop key.

        This method is intended to be called at epoch boundaries in the
        training loop.

        Returns:
            True if the stop signal has been received.
        """
        return self.event.is_set()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _listen(self) -> None:
        """Daemon target: continuously read keystrokes and detect the stop key.

        Platform‑specific I/O is abstracted via :meth:`_read_char`.
        """
        self._setup_terminal()
        try:
            while self._running and not self.event.is_set():
                ch = self._read_char()
                if ch is not None and ch == self.stop_key:
                    logger.warning(
                        "Stop key '%s' detected — training will terminate "
                        "gracefully at the next epoch boundary.",
                        self.stop_key,
                    )
                    self.event.set()
                    break
                # Tiny sleep to avoid busy‑waiting
                time.sleep(0.05)
        except Exception:
            logger.debug("Graceful-stop listener encountered an error.", exc_info=True)
        finally:
            self._cleanup_terminal()

    def _setup_terminal(self) -> None:
        """Place the terminal in cbreak mode on Unix so single keystrokes are
        available immediately, without breaking output newline formatting.
        No‑op on Windows (msvcrt handles this)."""
        if platform.system() == "Windows":
            return
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            self._old_term_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (ImportError, OSError, AttributeError, termios.error):
            logger.debug(
                "Could not set cbreak terminal mode — will use line‑buffered input.",
                exc_info=True,
            )
            self._old_term_settings = None

    def _cleanup_terminal(self) -> None:
        """Restore the original terminal attributes (Unix only)."""
        if self._old_term_settings is None:
            return
        try:
            import termios

            fd = sys.stdin.fileno()
            # Only restore if settings have changed
            current = termios.tcgetattr(fd)
            if current != self._old_term_settings:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term_settings)
        except Exception:
            logger.debug("Failed to restore terminal settings.", exc_info=True)
        finally:
            self._old_term_settings = None

    def _read_char(self) -> str | None:
        """Read a single character from stdin if available.

        Returns:
            The character as a string, or ``None`` if no input is ready.
        """
        try:
            if platform.system() == "Windows":
                return self._read_char_windows()
            return self._read_char_unix()
        except Exception:
            return None

    @staticmethod
    def _read_char_windows() -> str | None:
        """Windows single‑character read via msvcrt."""
        try:
            import msvcrt  # noqa: PLC0415  # Windows‑only

            if msvcrt.kbhit():  # type: ignore[attr-defined]
                ch = msvcrt.getch()  # type: ignore[attr-defined]
                # Decode bytes – handle both bytes and str
                if isinstance(ch, bytes):
                    ch = ch.decode("utf-8", errors="replace")
                return ch
            return None
        except ImportError:
            return None

    @staticmethod
    def _read_char_unix() -> str | None:
        """Unix single‑character read via select (non‑blocking)."""
        try:
            import select  # noqa: PLC0415

            if select.select([sys.stdin], [], [], 0.0) == ([sys.stdin], [], []):
                ch = sys.stdin.read(1)
                return ch if ch else None
            return None
        except Exception:
            return None


# ── Convenience factory ─────────────────────────────────────────────────────

_stop_instance: GracefulStop | None = None
"""Module‑level singleton so that :func:`requested` can be used without
explicitly managing a :class:`GracefulStop` instance."""


def setup_graceful_stop(stop_key: str = "q") -> GracefulStop:
    """Create, start, and return a module‑level :class:`GracefulStop` instance.

    Subsequent calls return the existing instance (singleton pattern).

    Args:
        stop_key: The key that triggers a graceful stop.

    Returns:
        The shared :class:`GracefulStop` instance.
    """
    global _stop_instance
    if _stop_instance is None:
        _stop_instance = GracefulStop(stop_key=stop_key)
        _stop_instance.start()
    return _stop_instance


def stop_requested() -> bool:
    """Check whether the stop key has been pressed (uses module‑level singleton).

    Returns:
        ``True`` if a graceful stop has been requested.
    """
    global _stop_instance
    if _stop_instance is None:
        return False
    return _stop_instance.stop_requested()


def teardown_graceful_stop() -> None:
    """Stop the module‑level listener and restore terminal."""
    global _stop_instance
    if _stop_instance is not None:
        _stop_instance.stop()
        _stop_instance = None

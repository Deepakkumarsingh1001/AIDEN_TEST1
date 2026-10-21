"""
Warehouse CoPilot 2.0 — Centralised logging + LangChain callback handler
=========================================================================

Usage:
    from tools.warehouse_logger import get_logger, WarehouseCallbackHandler

    logger = get_logger("warehouse.mymodule")
    logger.info("something happened")

    # Pass to the agent:
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(callbacks=[WarehouseCallbackHandler()])
    result = agent.invoke(inputs, config=config)

Log output:
    Console  — INFO and above  (synchronous)
    File     — DEBUG and above (async via QueueHandler — does not block request thread)
    Location — <project_root>/logs/warehouse_copilot.log
"""

import atexit
import json
import logging
import logging.handlers
import os
import queue
import time
from logging.handlers import RotatingFileHandler
from langchain_core.callbacks import BaseCallbackHandler

# ── Log directory ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(_LOG_DIR, "warehouse_copilot.log")

_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Async file handler (shared across all loggers) ────────────────────────────
# Requests enqueue log records in microseconds; a single background thread
# handles the actual RotatingFileHandler disk writes so they never block the
# Flask request thread.
_log_queue: queue.Queue = queue.Queue(-1)

_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_FMT)

_queue_listener = logging.handlers.QueueListener(
    _log_queue, _file_handler, respect_handler_level=True
)
_queue_listener.start()
atexit.register(_queue_listener.stop)

_queue_handler = logging.handlers.QueueHandler(_log_queue)
_queue_handler.setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger wired to console (sync) + rotating file (async)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Console — INFO+ (synchronous, negligible cost)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(_FMT)
        logger.addHandler(ch)

        # File — DEBUG+ via shared async queue (non-blocking)
        logger.addHandler(_queue_handler)

    return logger


# ── Module-level loggers used by the callback handler ────────────────────────
_tool_log = get_logger("warehouse.tool")
_llm_log = get_logger("warehouse.llm")
_chain_log = get_logger("warehouse.chain")


def _trim(text, limit: int = 300) -> str:
    s = str(text)
    return s[:limit] + "…" if len(s) > limit else s


# ── LangChain / LangGraph callback handler ────────────────────────────────────

class WarehouseCallbackHandler(BaseCallbackHandler):
    """Logs every tool call, LLM invocation, and chain step."""

    # ── Tool callbacks ────────────────────────────────────────────────────────

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        name = serialized.get("name", "unknown")
        _tool_log.info(f"START  {name:30s} | input: {_trim(input_str)}")

    def on_tool_end(self, output, **kwargs):
        result_type = "?"
        try:
            obj = json.loads(str(output)) if isinstance(output, str) else output
            result_type = obj.get("_result_type", "?")
            if "error" in obj:
                _tool_log.warning(f"END    result_type={result_type} | ERROR: {_trim(obj['error'])}")
                return
        except Exception:
            pass
        _tool_log.info(f"END    result_type={result_type} | output_len={len(str(output))}")

    def on_tool_error(self, error, **kwargs):
        _tool_log.error(f"ERROR  {error}")

    # ── LLM callbacks ─────────────────────────────────────────────────────────

    def on_chat_model_start(self, serialized: dict, messages, **kwargs):
        model = serialized.get("kwargs", {}).get("model_name") or serialized.get("name", "llm")
        msg_count = sum(len(batch) for batch in messages)
        _llm_log.info(f"START  model={model} | messages_in_batch={msg_count}")
        self._llm_start = time.time()

    def on_llm_end(self, response, **kwargs):
        elapsed = time.time() - getattr(self, "_llm_start", time.time())
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage", {})
        if usage:
            _llm_log.info(
                f"END    elapsed={elapsed:.2f}s | "
                f"prompt={usage.get('prompt_tokens','?')} "
                f"completion={usage.get('completion_tokens','?')} "
                f"total={usage.get('total_tokens','?')}"
            )
        else:
            _llm_log.info(f"END    elapsed={elapsed:.2f}s")

    def on_llm_error(self, error, **kwargs):
        _llm_log.error(f"ERROR  {error}")

    # ── Chain (graph node) callbacks ──────────────────────────────────────────

    def on_chain_start(self, serialized, inputs: dict, **kwargs):
        if not serialized:
            return
        name = serialized.get("name") or (serialized.get("id") or ["?"])[-1]
        keys = list(inputs.keys()) if isinstance(inputs, dict) else []
        _chain_log.debug("START  %s | input_keys=%s", name, keys)

    def on_chain_end(self, outputs, **kwargs):
        if not isinstance(outputs, dict):
            return
        _chain_log.debug("END    output_keys=%s", list(outputs.keys()))

    def on_chain_error(self, error, **kwargs):
        _chain_log.error(f"ERROR  {error}")

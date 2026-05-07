import sys
import os
from loguru import logger
from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

sys.path.insert(0, os.path.dirname(__file__))

# ── Structured Logging (Loguru) ──────────────────────────────────────────
# Replaces bare print() with structured JSON logs that carry trace context.
# Logs are written to both stderr (colored, human-readable) and
# rag_assistant.log (JSON, machine-parseable for log aggregation).
logger.remove()  # remove default handler

# Human-friendly console output
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)

# JSON-structured file log for observability pipeline
logger.add(
    "rag_assistant.log",
    format="{{ \"time\": \"{time}\", \"level\": \"{level}\", \"name\": \"{name}\", \"function\": \"{function}\", \"line\": {line}, \"message\": \"{message}\" }}",
    level="DEBUG",
    rotation="50 MB",
    retention=3,
)

logger.info("Loguru initialized — structured logging active")

if __name__ == "__main__":
    demo = create_gradio_ui()
    logger.info("Launching RAG Assistant UI...")
    demo.launch(css=custom_css)
    logger.success("RAG Assistant UI is running")

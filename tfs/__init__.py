"""
tfs — Transformers From Scratch reference implementation.

Every module here is a *working, tested* version of something the course asks
you to build. Use it to check your own implementation, not to skip writing one.

    from tfs.tokenizer import BPETokenizer
    from tfs.model import Config, Model
    from tfs.train import train
    from tfs.dpo import dpo_loss

Run the test suite with:

    python -m pytest tests/ -v
    # or, with no pytest installed:
    python tests/test_tfs.py
"""

__version__ = "1.0.0"

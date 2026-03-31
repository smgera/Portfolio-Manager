"""Centralized logging configuration: root or named logger writing to console and report file."""
from __future__ import annotations

import logging
import os
from typing import Tuple


def setup_root_logger(output_filename: str, report_dir: str = "reports", level: int = logging.INFO) -> str:
    """Configure the root logger to write to console and a report file.
    Returns:
        Full path to the report file
    Side Effects:
        - Creates report_dir directory if it does not exist
        - Clears existing root logger handlers to avoid duplicates
        - Adds FileHandler (writing to report file) and StreamHandler to root logger
    """
    os.makedirs(report_dir, exist_ok=True)
    report_filename = Path(report_dir) / f"{output_filename}_report.txt"

    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicates (especially in notebooks / reruns)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.setLevel(level)
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(report_filename, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return report_filename


def setup_named_logger(logger_name: str,
                       output_filename: str,
                       report_dir: str = "reports",
                       level: int = logging.INFO,
                       propagate: bool = False) -> logging.Logger:
    """Configure and return a named logger that logs to console and a report file.
    Side Effects:
        - Creates report_dir directory if it does not exist
        - Clears existing handlers on this named logger to avoid duplicates
        - Adds FileHandler (report file) and StreamHandler to the named logger
    """
    os.makedirs(report_dir, exist_ok=True)
    report_filename = Path(report_dir) / f"{output_filename}_report.txt"

    logger = logging.getLogger(logger_name)
    # Clear existing handlers to avoid duplicate logs across reruns
    for h in list(logger.handlers):
        logger.removeHandler(h)

    logger.setLevel(level)
    logger.propagate = propagate
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(report_filename, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

"""
Setup script for Portfolio Manager.

This script allows installation of the Portfolio Manager as an editable package
in your Python environment.
"""

from setuptools import setup, find_packages
import os

# Read requirements from pyproject.toml or use hardcoded dependencies
dependencies = [
    "pandas",
    "numpy",
    "yfinance",
    "exchange-calendars",
    "pyarrow",
    "requests",
    "python-dotenv",
]

# Optional dependencies for full functionality
extras_require = {
    "dev": [
        "pytest",
        "mypy",
        "ruff",
        "black",
    ],
    "ui": [
        "flet",
        "duckdb",
        "quantstats",
        "pyportfolioopt",
        "matplotlib",
    ],
    "all": [
        "pytest",
        "mypy",
        "ruff",
        "black",
        "flet",
        "duckdb",
        "quantstats",
        "pyportfolioopt",
        "matplotlib",
    ],
}

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Portfolio-Manager",
    version="2.0.0",
    author="Steve Gerads",
    author_email="",
    description="Portfolio management and performance tracking tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smgera/Portfolio-Manager",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.11",
    install_requires=dependencies,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "portfolio-manager=Monitor.portfolio_performance:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)

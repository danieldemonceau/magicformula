"""Setup configuration for magicformula package."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="magicformula",
    version="1.0.0",
    description="Investment Strategy Analysis Tool (Magic Formula, Acquirer's Multiple, DCA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.28",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
            "types-python-dateutil>=2.8.19",
        ],
    },
    entry_points={
        "console_scripts": [
            "magicformula=scripts.run_analysis:main",
        ],
    },
)


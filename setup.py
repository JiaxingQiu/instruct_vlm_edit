from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent
reqs_path = root / "requirements.txt"
install_requires = reqs_path.read_text().splitlines() if reqs_path.exists() else []

setup(
    name="revlm",
    version="0.1.0",
    description="Rationale Editing for VLM",
    packages=find_packages(include=["revlm", "revlm.*"]),
    python_requires=">=3.9",
    install_requires=install_requires,
)
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dspy-elo",
    version="0.1.0",
    author="Tom Doerr",
    author_email="",
    description="An Elo-based metric function for DSPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tom-doerr/dspy_elo",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "dspy",
        "numpy",
    ],
)

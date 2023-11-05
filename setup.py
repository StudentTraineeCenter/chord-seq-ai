from setuptools import setup

setup(
    name="src",
    version="0.1",
    license="MIT",
    url="https://github.com/StudentTraineeCenter/chord-seq-ai",
    description="Source code for chord-seq-ai notebooks",
    author="Petr Ivan",
    packages=["src"],
    install_requires=[
        "numpy",
        "pandas",
        "music21",
        "regex",
        "torch",
    ],
)

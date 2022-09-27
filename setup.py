from setuptools import setup, find_packages

with open("README.md", "r") as source:
    long_description = source.read()

setup(
    name="foldingdiff",
    packages=find_packages(),
    description="Diffusion for protein backbone generation using internal angles",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

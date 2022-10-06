from setuptools import setup, find_packages

with open("README.md", "r") as source:
    long_description = source.read()

setup(
    name="foldingdiff",
    author="Kevin Wu",
    packages=find_packages(),
    include_package_data=True,
    description="Diffusion for protein backbone generation using internal angles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/foldingdiff",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    version="0.0.1",
)

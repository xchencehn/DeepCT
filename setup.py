from setuptools import setup, find_packages

setup(
    name="deepct",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="DeepCT: internal signal tomography for Transformer models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "loguru",
    ],
    python_requires=">=3.8",
)
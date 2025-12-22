from setuptools import setup, find_packages

setup(
    name="deepct",
    version="0.1.0",
    author="xchencehn",
    author_email="xchencehn@gmail.com",
    description="DeepCT: internal signal tomography for Transformer models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt", "r", encoding="utf-8")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.8",
)
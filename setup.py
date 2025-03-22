from setuptools import setup, find_packages

setup(
    name="adasplash",
    version="0.1.1b",
    author="Nuno Gonçalves, Marcos Treviso",
    author_email="marcosvtreviso@gmail.com",
    description="AdaSplash: Efficient Adaptive Sparse Attention in Triton",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deep-spin/adasplash",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "triton",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
            "entmax"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
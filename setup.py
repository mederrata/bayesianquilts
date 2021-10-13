import setuptools

"""
https://packaging.python.org/tutorials/packaging-projects/
"""

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="bayesianquilts",
    version="0.0.6",
    author="mederrata",
    author_email="info@mederrata.com",
    description="Quilting a Bayesian Hierarchical model to mimic relu neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mederrata/bayesianquilts",
    packages=setuptools.find_packages(
        exclude=["*.md", "aws", "bin/*.sh", "design_docs", "tools/"]
    ),
    include_package_data=True,
    package_data={
        "bayesianquilts": [
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Closed",
        "Operating System :: Linux",
    ],
    scripts=[
    ],
    python_requires='>=3.6',
    install_requires=[
        'dill>=0.3.1.1',
        'matplotlib>=3.1',
        'arviz>=0.10.0',
        'numpy~=1.19.2',
        'tensorflow~=2.5.0',
        'tensorflow-probability~=0.13.0',
        'tensorflow-addons~=0.13.0',
        'jax',
        'jaxlib',
        'natsort',
        'tqdm'
    ]
)

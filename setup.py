import setuptools

"""
https://packaging.python.org/tutorials/packaging-projects/
"""

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="bayesianquilts",
    version="0.1.2",
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
        "License :: MIT",
        "Operating System :: Linux",
    ],
    scripts=[
    ],
    python_requires='>=3.8',
    install_requires=[
        'dill',
        'matplotlib',
        'arviz',
        'numpy',
        'pandas',
        # We need to check for direct depends or we can delete scipy.
        'scipy',
        'protobuf~=3.19.0',
        'tensorflow==2.10.1',
        'tensorflow-probability==0.18.0',
        'tensorflow-addons==0.18.0',
        'jax',
        'natsort',
        'tqdm',
    ]
)

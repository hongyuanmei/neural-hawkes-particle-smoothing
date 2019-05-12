from setuptools import setup, find_packages

setup(
    name="neural-hawkes-particle-smoothing",
    version="1.0",
    packages=find_packages(),
    install_requires=['torch', 'numpy', 'matplotlib']
)

from setuptools import setup, find_packages

setup(
    name='torchexp',
    version='0.1.0',
    packages=find_packages(exclude=['docs', 'tests', 'examples']),
    install_requires=[
        'numpy >= 1.16.0',
        'torch >= 1.1',
        'gin-config >= 0.1.4',
    ],
)

from setuptools import find_packages, setup

setup(
    name='HAP_py',
    packages=find_packages(include=['HAP.*']),
    install_requires=[
        'networkx==2.8.8',
        'scipy',
        'numpy',
        'pyyaml',
    ],
    version='0.1.0',
    description='Hausdorff Approximation Generator (supports multiple HAs)',
    author='Fouad (Fred) Sukkar',
)
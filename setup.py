from setuptools import setup, find_packages

setup(
    name='planimals',
    version='0.1',
    packages=find_packages(where="src"),
    install_requires=[
        "python-ucto",
        "regex",
        "lxml",
        "tqdm",
        "pandas",
        "folia",        
        "langdetect"
    ],

    package_dir={'': 'src'},
)
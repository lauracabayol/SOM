import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# This requires PyTorch, which cannot be automatically  installed
# using pip.

setup(
    name = "SOM",
    version = "1.0.0",
    author = "Laura Cabayol",
    author_email = "lauracabayol@gmail.com",
    description = ("Self Organising Map"),
    keywords = "astronomy",
    url = "https://github.com/lauracabayol/SOM",
    license="GPLv3",
    packages=['SOM'],
    install_requires=['numpy', 'pandas','scipy',],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)

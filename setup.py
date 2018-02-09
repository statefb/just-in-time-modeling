# -*- coding: utf-8 -*-
import os
from setuptools import setup
from setuptools import find_packages

# meta info
NAME = "jitmodel"
VERSION = "0.0.1"
AUTHOR = "Takehiro Suzuki"
AUTHOR_EMAIL = ""
URL = ""
DESCRIPTION = ""
LICENSE = ""

# if not os.path.exists('README.txt'):
#     os.system("pandoc -o README.txt README.md")
# LONG_DESCRIPTION = open('README.txt').read()
LONG_DESCRIPTION = ""

def main():

    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        zip_safe=False,
        include_package_data=True,
        packages=find_packages(),
        install_requires=[

        ],
        dependency_links = [

        ],
        tests_require=[],
        setup_requires=[],
        license=LICENSE,
        classifiers = [
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
        ]
    )


if __name__ == '__main__':
    main()

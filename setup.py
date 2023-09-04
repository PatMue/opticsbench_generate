# Aug2023 (c) Patrick Müller  - "opticsbenchgen"

import os
import sys

__install_requirements__ = False # True 


if __install_requirements__:
    __requirements__ = 'requirements_cpu.txt'
else:
	__requirements__ = None


from setuptools import find_packages,setup 
from pathlib import Path

setup(
    name='opticsbenchgen',
    version='0.1',
    description="OpticsBench Generator (initial)",
    long_description='',
    license="GNU General public license GNU GPLv3, Patrick Müller (c) 2020-2023",
    classifiers=[
        "Development Status :: 2 - Development",
        "Intended Audience :: Developers (Confidential)",
        "Intended Audience :: Researchers (Confidential)",
        "License :: Non-commercial (not yet specified) (do not distribute, confidential)",
        "Topic :: Scientific",
        "Programming Language :: Python :: 3.8"
    ],
    url="",
    project_urls={
        "Documentation": "",
        "Source": "",
        "Changelog": "",
    },
    author="Patrick Müller",
    author_email="patrick.mueller.de@gmail.com",
    packages=["opticsbenchgen/PSF","opticsbenchgen/utils","opticsbenchgen"],
    package_data={},
    entry_points={},
    zip_safe=False,
    python_requires='>=3.8.3',
    install_requires=Path(__requirements__).read_text(encoding="utf-8").split("\n") if \
       __install_requirements__ else [""]
)
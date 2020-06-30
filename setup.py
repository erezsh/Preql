import re
from setuptools import find_packages, setup

__version__ ,= re.findall('__version__ = "(.*)"', open('preql/__init__.py').read())

setup(
    name = "preql",
    version = __version__,
    packages = ['preql'],

    requires = [],
    install_requires = ['lark-parser>=0.8.8', 'runtype>=0.1.4', 'tabulate', 'dsnparse', 'tqdm', 'prompt-toolkit', 'pygments', 'psycopg2'],

    package_data = {'': ['*.md', '*.lark', '*.pql']},

    test_suite = 'tests.__main__',

    # metadata for upload to PyPI
    author = "Erez Shinan",
    author_email = "erezshin@gmail.com",
    description = "Pretty Query Language",
    license = "MIT",
    keywords = "Preql SQL",
    url = "",
    # scripts=['bin/preql'],
    entry_points={'console_scripts': ['preql=preql.__main__:main'], },
    long_description=''',
    "Pretty Query Language",
''',

    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)


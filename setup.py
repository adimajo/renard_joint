"""
A setuptools based setup module.
"""
import os
import re
import codecs
# Always prefer setuptools over distutils
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


def long_description():
    with open(os.path.join(here, 'README.md')) as f:
        long_description = f.read()
    return long_description


def find_version(file_path, file_name):
    """
    Get the version from __init__.py file

    Parameters
    ----------
    file_path: path of this file
    file_name: which python file to search for the version

    Returns
    -------
    version
    """
    with codecs.open(os.path.join(file_path, file_name), 'r') as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('requirements.txt') as fp:
    install_requires = [x.split("/")[-1] for x in fp.read().splitlines()[1:]]


if __name__ == "__main__":
    # Arguments marked as "Required" below must be included for upload to PyPI.
    # Fields marked as "Optional" may be commented out.
    setup(
        setup_requires=["wheel"],
        # This is the name of your project. The first time you publish this
        # package, this name will be registered for you. It will determine how
        # users can install this project, e.g.:
        #
        # $ pip install sampleproject
        #
        # And where it will live on PyPI: https://pypi.org/project/sampleproject/
        #
        # There are some restrictions on what makes a valid project name
        # specification here:
        # https://packaging.python.org/specifications/core-metadata/#name
        name="renard_joint",  # Required
        # Versions should comply with PEP 440:
        # https://www.python.org/dev/peps/pep-0440/
        #
        # For a discussion on single-sourcing the version across setup.py and the
        # project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version=find_version(here, "renard_joint/__init__.py"),  # Required
        # This is a one-line description or tagline of what your project does. This
        # corresponds to the "Summary" metadata field:
        # https://packaging.python.org/specifications/core-metadata/#summary
        description="Joint entity and relation extraction",  # Required
        # This is an optional longer description of your project that represents
        # the body of text which users will see when they visit PyPI.
        #
        # Often, this is the same as your README, so you can just read it in from
        # that file directly (as we have already done above)
        #
        # This field corresponds to the "Description" metadata field:
        # https://packaging.python.org/specifications/core-metadata/#description-optional
        long_description=long_description(),
        long_description_content_type="text/markdown",  # Optional
        # This should be a valid link to your project's main homepage.
        #
        # This field corresponds to the "Home-Page" metadata field:
        # https://packaging.python.org/specifications/core-metadata/#home-page-optional
        url="https://gitlab.ca.cib/RCI/GRP/GRO_PoVs/datahavesting",
        author="Adrien Ehrhardt",
        author_email="adrien.ehrhardt@credit-agricole-sa.fr",
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        keywords="nlp text entity recognition relation extraction",
        packages=find_packages(exclude=["contrib", "docs", "tests", "examples", "venv"]),
        install_requires=install_requires,
        test_suite="pytest-runner",
        tests_require=[],
        # If there are data files included in your packages that need to be
        # installed, specify them here.
        #
        # If using Python 2.6 or earlier, then these have to be included in
        # MANIFEST.in as well.
        # package_data={  # Optional
        #    'sample': ['package_data.dat'],
        # },
        # Although 'package_data' is the preferred approach, in some case you may
        # need to place data files outside of your packages. See:
        # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
        #
        # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
        # data_files=[('my_data', ['data/data_file'])],  # Optional
        # To provide executable _scripts, use entry points in preference to the
        # "_scripts" keyword. Entry points provide cross-platform support and allow
        # `pip` to create the appropriate form of executable for the target
        # platform.
        #
        # For example, the following would provide a command called `sample` which
        # executes the function `main` from this package when invoked:
        # entry_points={  # Optional
        #    'console_scripts': [
        #        'sample=sample:main',
        #    ],
        # },
        entry_points={
            'console_scripts': [
                'pipeline=renard_joint._scripts.pipeline:main',
                'spert=renard_joint._scripts.spert:main',
            ],
        }
    )

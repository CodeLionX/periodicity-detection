import glob
import os
import shutil
import sys
from pathlib import Path

from setuptools import setup, find_packages, Command


HERE = Path(os.path.dirname(__file__)).absolute()
# get __version__ from periodicity_detection/_version.py
with open(HERE / "periodicity_detection" / "_version.py") as fh:
    exec(fh.read())
VERSION: str = __version__  # noqa
README = (HERE / "README.md").read_text(encoding="UTF-8")
DOC_NAME = "periodicity-detection"
PYTHON_NAME = "periodicity_detection"
with open(HERE / "requirements.txt") as fh:
    REQUIRED = fh.read().splitlines()
with open(HERE / "requirements.dev") as fh:
    DEV_REQUIRED = [
        dep
        for dep in fh.read().splitlines()
        if dep and not dep.startswith("-") and not dep.startswith("#")
    ]


class PyTestCommand(Command):
    description = f"run PyTest for {DOC_NAME}"
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        import pytest
        from pytest import ExitCode

        exit_code = pytest.main(
            [
                "--cov-report=term",
                "--cov-report=xml:coverage.xml",
                f"--cov={PYTHON_NAME}",
                "tests",
            ]
        )
        if exit_code == ExitCode.TESTS_FAILED:
            raise ValueError("Tests failed!")
        elif exit_code == ExitCode.INTERRUPTED:
            raise ValueError("pytest was interrupted!")
        elif exit_code == ExitCode.INTERNAL_ERROR:
            raise ValueError("pytest internal error!")
        elif exit_code == ExitCode.USAGE_ERROR:
            raise ValueError("Pytest was not correctly used!")
        elif exit_code == ExitCode.NO_TESTS_COLLECTED:
            raise ValueError("No tests found!")
        # else: everything is fine


class MyPyCheckCommand(Command):
    description = f"run MyPy for {DOC_NAME}; performs static type checking"
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from mypy.main import main as mypy

        args = ["--pretty", PYTHON_NAME, "tests"]
        mypy(None, stdout=sys.stdout, stderr=sys.stderr, args=args)


class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        files = [".coverage*", "coverage.xml"]
        dirs = [
            "build",
            "dist",
            "*.egg-info",
            "**/__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            "**/.ipynb_checkpoints",
        ]
        for d in dirs:
            for filename in glob.glob(d):
                shutil.rmtree(filename, ignore_errors=True)

        for f in files:
            for filename in glob.glob(f):
                try:
                    os.remove(filename)
                except OSError:
                    pass


if __name__ == "__main__":
    setup(
        name=PYTHON_NAME,
        version=VERSION,
        description="Detect the dominant period in univariate, equidistant "
        "time series data.",
        long_description=README,
        long_description_content_type="text/markdown",
        author="Sebastian Schmidl",
        author_email="sebastian.schmidl@hpi.de",
        url="https://github.com/CodeLionX/periodicity_detection",
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Typing :: Typed",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
        ],
        packages=find_packages(exclude=("tests", "tests.*")),
        package_data={"periodicity_detection": ["py.typed"]},
        install_requires=REQUIRED,
        extras_require={
            "dev": DEV_REQUIRED,
        },
        python_requires=">=3.7",
        test_suite="tests",
        cmdclass={
            "test": PyTestCommand,
            "typecheck": MyPyCheckCommand,
            "clean": CleanCommand,
        },
        zip_safe=False,
        entry_points={
            "console_scripts": ["periodicity=periodicity_detection.__main__:cli"]
        },
    )

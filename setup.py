from pathlib import Path

from setuptools import find_packages, setup

import versioneer

here = Path(__file__).expanduser().absolute().parent
readme = here / "README.md"
long_description = readme.read_text(encoding="utf-8")


setup(
    name="quantkit",
    version=versioneer.get_version(),
    description="Quantitative functions for finance analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asdf8601/quantkit",
    author="mmngreco",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.5, <4",
    install_requires=["numpy", "pandas"],
)

import pathlib
import setuptools

# Based on https://shunsvineyard.info/2019/12/23/using-git-submodule-and-develop-mode-to-manage-python-projects/

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setuptools.setup(
    name="sssl",
    version="0.0.2",
    description="",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Ruben Cartuyvels",
    author_email="ruben.cartuyvels@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3"
)

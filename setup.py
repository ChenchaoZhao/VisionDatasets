from setuptools import find_packages, setup

from torchvision_extra import __version__

# load readme
with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="torchvision-extra",
    version=__version__,
    author="Chenchao Zhao",
    author_email="chenchao.zhao@gmail.com",
    description="A collection of vision datasets as an addition to torchvision datasets",
    packages=find_packages(exclude=["tests"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["torch", "numpy", "torchvision", "pillow"],
    license="MIT",
)

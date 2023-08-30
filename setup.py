from setuptools import setup


def read_requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()

setup(
    name="icl",
    version="0.0.0",
    packages=["icl"],
    # license="LICENSE",
    description="Singular Learning Theory for In-Context Learning",
    long_description=open("README.md").read(),
    install_requires=read_requirements(),
    extras_require={"dev": ["pytest", "mypy", "pytest-cov"]},
)
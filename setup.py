from setuptools import find_packages, setup


def read_requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()

setup(
    name="icl",
    version="0.0.0",
    description="The Developmental Landscape of In-Context Learning",
    long_description=open("README.md").read(),
    install_requires=read_requirements(),
    extras_require={"dev": ["pytest", "torch_testing"]},
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)



from setuptools import setup


def readme():
    try:
        with open("README.rst") as f:
            return f.read()
    except IOError:
        return ""


setup(
    name="empiricaldist",
    version="0.7.7",
    author="Allen B. Downey",
    author_email="downey@allendowney.com",
    packages=["empiricaldist"],
    install_requires=["matplotlib", "numpy", "pandas", "scipy"],
    python_requires=">=3.7",
    url="https://github.com/AllenDowney/empiricaldist",
    license="BSD-3-Clause",
    description="Python library that represents empirical distributions.",
    long_description=readme(),
)

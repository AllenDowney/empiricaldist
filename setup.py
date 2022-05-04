from distutils.core import setup


def readme():
    try:
        with open("README.rst") as f:
            return f.read()
    except IOError:
        return ""


setup(
    name='empiricaldist',
    version='0.6.5',
    author='Allen B. Downey',
    author_email='downey@allendowney.com',
    packages=['empiricaldist'],
    url='http://github.com/AllenDowney/empiricaldist',
    license='LICENSE',
    description='Python library that represents empirical distributions.',
    long_description=readme(),
)

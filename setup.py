from distutils.core import setup

def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except IOError:
        return ''


setup(
    name='empyrical_dist',
    version='0.1.0',
    author='Allen B. Downey',
    author_email='downey@allendowney.com',
    packages=['empyrical_dist'],
    url='http://github.com/AllenDowney/EmpyricalDistributions',
    license='LICENSE',
    description='Python library that represents empirical distribution functions.',
    long_description=readme(),
)

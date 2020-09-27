from setuptools import setup



def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


setup(
    name='GPBlocks',
    version='0.1.0',
    author='Thomas Pinder',
    author_email='t.pinder2@lancaster.ac.uk',
    packages=['gpblocks'], #, 'package_name.test'],
    # scripts=['bin/script1', 'bin/script2'],
    # url='http://pypi.python.org/pypi/PackageName/',
    license='LICENSE',
    description='Building blocks of Gaussian processes. Made to enable rapid development for researchers.',
    long_description=open('README.md').read(),
    install_requires=parse_requirements_file("requirements.txt"),
)

from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='xylo',
    version='0.0.1',
    description='Neuromorphic bird sound detection.',
    long_description=readme,
    author='Daniel Müller-Komorowska',
    author_email='danielmuellermsc@gmail.com',
    url='https://github.com/danielmk/xylo',
    license=license,
    packages=['xylo'],
    install_requires=[
        'rockpool[xylo, tests, docs]',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'librosa',
        'parquetdb'],)

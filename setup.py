from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='project',
    version='0.0.1',
    description='Template for Python based neuroscience research.',
    long_description=readme,
    author='Daniel Müller-Komorowska',
    author_email='danielmuellermsc@gmail.com',
    url='https://github.com/danielmk/template',
    license=license,
    packages=['project'],
    install_requires=[
          'sbi==0.24.0',
          'numpy',
          'pandas',
          'matplotlib',
          'scipy',],)

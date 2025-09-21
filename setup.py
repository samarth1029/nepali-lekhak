from setuptools import find_packages, setup
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Nepali-centric LLM for Nepali NLP tasks.',
    author='Samarth Mishra',
    license='MIT',
    install_requires=requirements,
)

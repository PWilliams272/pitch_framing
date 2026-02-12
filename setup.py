from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pitch_framing',
    version='0.1.0',
    description='Pitch framing analysis using video and object detection',
    author='Peter Williams',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
)

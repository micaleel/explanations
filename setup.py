"""
Explanation Installer

For development:
    `python setup.py develop` or `pip install -e .`
"""

from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    requirements_path = os.path.join(here, "requirements.txt")
    requirements = open(requirements_path, 'r').readlines()

    setup(
        name='Explanations',
        description='Opinionated Explanations Framework',
        packages=find_packages(),
        version='0.1.0',
        author='Khalil Muhammad',
        author_email='khalil.muhammad@insight-centre.org',
        license='All Rights Reserved',
        install_requires=requirements,
        entry_points={
            'console_scripts': [
                'explain = explanations.explain:main',
            ]
        },
    )

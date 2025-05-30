from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace('\n', '') for requirement in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='NutriScan',
    version='1.0.0',
    author='Ritvik Sharma, Prachi Tayshete, Tanmay Sinkar',
    author_email='ritvik.sharma22@spit.ac.in, prachi.tayshete22@spit.ac.in, tanmay.sinkar22@spit.ac.in',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
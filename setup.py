from setuptools import find_packages,setup
from typing import List


hypen_e_dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this return list of requiremets
    '''
    requiremets = []
    with open(file_path) as f:
        requiremets= f.readlines()
        requiremets=[reg.replace('\n', '') for reg in requiremets]

        if hypen_e_dot in requiremets:
            requiremets.remove(hypen_e_dot)

    return requiremets


setup(
    name='mlproject',
    version='0.0.1',
    author='Dejene',
    author_email='hailudjj@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
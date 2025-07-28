import re 
from pathlib import Path 

from setuptools import setup, find_packages

FILE = Path(__file__).resolve()
PARENT = FILE.parent 
README = (PARENT / 'README.md').read_text(encoding='utf-8')

def get_version():
    file = PARENT / 'visionsuite/__init__.py'
    
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]

def parse_requirements(file_path: Path):
    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        
        if line and not line.startswith('#'):
            requirements.append(line.split("#")[0].strip())
    
    return requirements 

try:
    setup(
        name='visionsuite',
        version='{{VERSION_PLACEHOLDER}}',
        python_requires='>=3.9',
        description=('Template for Python Athena'),
        long_description=README,
        long_description_content_type='text/markdown',
        # packages=['athena'] + [str(x) for x in Path('athena').rglob('*/') if x.is_dir() and '__' not in str(x)],
        packages=find_packages(exclude=[]),
        package_data={
            '': ['*.yaml', '*.json'], },
        include_package_data=True,
        install_requires=parse_requirements(PARENT / 'requirements.txt') + ["nvidia-ml-py3"],
    )
except:
    setup(
        name='visionsuite',
        version=get_version(),
        python_requires='>=3.9',
        description=('Template for Python Athena'),
        long_description=README,
        long_description_content_type='text/markdown',
        # packages=['athena'] + [str(x) for x in Path('athena').rglob('*/') if x.is_dir() and '__' not in str(x)],
        packages=find_packages(exclude=[]),
        package_data={
            '': ['*.yaml', '*.json'], },
        include_package_data=True,
        # install_requires=["nvidia-ml-py3"],
        install_requires=parse_requirements(PARENT / 'requirements.txt') + ["nvidia-ml-py3"],
    )
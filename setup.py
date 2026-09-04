import os
from setuptools import setup, find_packages

# `lifesim/__init__.py` keeps LIFEsim's own version. See lifesim/UPSTREAM.md.
__version__ = '0.1.0'

def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding='utf-8') as fp:
        return fp.read()

setup(
    name='exoplanet-survey-simulator',
    version=__version__,
    description='Detectability of potentially habitable exoplanets around M dwarfs '
                'for the LIFE and HWO mission concepts',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author='Annika Salmi',
    author_email='annikaksalmi@gmail.com',
    url='https://github.com/annikasalmi/exoplanet-survey-simulator',
    # `lifesim/` is listed by hand so the inherited tree needs no added
    # __init__.py files. It is frozen at a2b8eeb, so the list will not drift.
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*', 'data', 'data.*',
                                    'output', 'output.*', 'lifesim', 'lifesim.*',
                                    '*.data', '*.data.*'])
             + ['lifesim',
                'lifesim.core',
                'lifesim.gui',
                'lifesim.instrument',
                'lifesim.optimize',
                'lifesim.util',
                'telescopes',
                'telescopes.kepler',
                'telescopes.tess',
                'telescopes.hwo',
                'telescopes.rv'],
    include_package_data=True,
    install_requires=['alphashape',
                      'astropy>=5.2.1',
                      'GitPython>=3.1.32',
                      'h5py',
                      'matplotlib>=3.7.0',
                      'numpy>=1.24.2',
                      'pandas>=1.5.3',
                      'PyQt5>=5.15.4,<6',
                      'pyyaml',
                      'requests',
                      'scipy>=1.7.0',
                      'spectres',
                      'tables>=3.8.0',
                      'tqdm>=4.64.1'],
    extras_require={'test': ['pytest>=6.0.0', 'pytest-cov>=2.10.0']},
    license='GPLv3',
    zip_safe=False,
    keywords='exoplanets astronomy LIFE HWO habitability',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ]
)

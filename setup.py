from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="soil-moisture-analyzer",
    version="0.1.0",
    author="Kyle Jones",
    author_email="KyleTJones@gmail.com",
    description="A Python package for processing and analyzing AMSR2 LPRM soil moisture data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kylejones200/soil-moisture-analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'scipy>=1.7.0',
        'netCDF4>=1.5.0',
        'cartopy>=0.19.0',
        'seaborn>=0.11.0',
    ],
    entry_points={
        'console_scripts': [
            'soilmoisture-analyze=soilmoisture.analyze:main',
            'soilmoisture-visualize=soilmoisture.visualize:main',
        ],
    },
)

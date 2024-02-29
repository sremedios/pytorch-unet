from setuptools import setup

version = '0.1.0'

with open('README.md') as readme:
    long_desc = readme.read()

setup(
    name='pytorch-unet',
    description='PyTorch Implementation of U-Net',
    author='Shuo Han',
    author_email='shan50@jhu.edu',
    version=version,
    packages=['pytorch_unet'],
    license='GPLv3',
    python_requires='>=3.7.10',
    long_description=long_desc,
    install_requires=[
        'torch>=1.8.1',
        'radifox-utils'
    ],
    long_description_content_type='text/markdown',
    url='https://github.com/shuohan/pytorch-unet.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent'
    ]
)

from setuptools import setup

setup(
    name='keras-tcn',
    version='4.0.0',
    description='Keras TCN',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['tcn'],
    install_requires=[
        'numpy', 'tensorflow', 'tensorflow_addons'
    ]
)

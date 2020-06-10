import setuptools


INSTALL_REQUIRES = [
    'torch>=1.5.0',
    'numpy',

]
EXTRAS_REQUIRE = {
    'dev': ['pytest'],
}

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='deepl',
    version='0.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.6',
)

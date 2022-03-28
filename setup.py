import setuptools
import os
import sys
import glob
import contextlib
from Cython.Build import cythonize


MOD_NAMES = [
    "models.bpe",
]

EXTENSIONS = [
    setuptools.Extension(
        "deepl.models.avg_ctx",
        [
            "deepl/models/avg_ctx.pyx",
            "deepl/src/avg_ctx.cpp",
        ],
        extra_compile_args=["-std=c++11", "-pthread", "-O2"],
        language="c++",
    )
]


def clean(path):
    for name in MOD_NAMES:
        name = name.replace(".", "/")
        for ext in [".so", ".html", ".cpp", ".c"]:
            file_path = glob.glob(os.path.join(path, name) + '*' + ext)
            for fp in file_path:
                os.unlink(fp)


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


INSTALL_REQUIRES = [
    'torch>=1.5.0',
    'numpy',
    'scipy'

]
EXTRAS_REQUIRE = {
    'dev': ['pytest', 'tqdm'],
}

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='deepl',
    version='1.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.6',
    ext_modules=cythonize(EXTENSIONS),
)

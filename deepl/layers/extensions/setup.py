from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='multi_head_attention',
      ext_modules=[cpp_extension.CppExtension(
            name='multi_head_attention',
            sources=['multi_head_attention.cpp'],
            extra_cflags=['-O3'],
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

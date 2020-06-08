from setuptools import setup, find_packages

setup(
  name = 'axial_positional_embedding',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'Axial Positional Embedding',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/axial-positional-embedding',
  keywords = ['transformers', 'artificial intelligence'],
  install_requires=[
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)
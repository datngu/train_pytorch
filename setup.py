from setuptools import setup, find_packages

setup(
  name = 'train_pytorch',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Simple trainer for pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Dat T Nguyen',
  author_email = 'ndat@utexas.edu',
  url = 'https://github.com/datngu/train_pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'pytorch'    
  ],
  install_requires=[
    'tqdm',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
  ],
)
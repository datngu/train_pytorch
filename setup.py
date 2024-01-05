from setuptools import setup, find_packages

# Read the content of README.md
with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()


setup(
  name = 'train_pytorch',
  packages = find_packages(),
  version = '0.0.3',
  license='MIT',
  description = 'Simple trainer for pytorch.',
  long_description=long_description,
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
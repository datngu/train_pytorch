# test local
pip install .

## buid
python setup.py sdist bdist_wheel

## upload
twine upload dist/*
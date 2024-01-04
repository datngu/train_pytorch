# test local
pip install .

##

python setup.py sdist bdist_wheel

twine upload dist/*
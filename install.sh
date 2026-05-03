rm -rf build
pip uninstall lblConvert -y
rm -rf dist
python -m build -sw -nx
pip install dist/*.whl
rm -rf log

#twine upload dist/*

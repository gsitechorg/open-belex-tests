source belex-venv/bin/activate
pip install pytest-xdist
pip install hypothesis
date
pytest -n 12 -vv -s
date

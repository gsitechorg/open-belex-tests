source ../belex-tests/belex-venv/bin/activate
echo "BE SURE TO MANUALLY SET import ipdb; ipdb.set_trace()"
echo "IN YOUR belex_apl FUNCTIONS AT THE POINT WHERE YOU"
echo "WISH TO START DEBUGGING."
pip install pudb
pip install ipdb
# OLD COMMAND NO LONGER WORKS AS OF 20 MAR 2022
# python -mipdb $(type -p pytest) -vv -s tests/test_belex_game_of_life.py::test_game_of_life_tutorial

pytest -vv -s tests/test_belex_game_of_life.py::test_game_of_life_tutorial --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb

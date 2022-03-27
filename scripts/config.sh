python -m venv venv
source venv/bin/activate
git submodule add -b main https://github.com/jsego/pacman-contest.git
cd pacman-contest/
pip install -e .
cd ..
pip install -r requirements.txt
# deactivate

# Setup

```bash
# create venv
sudo apt install python3-pip -y
sudo apt install python3-venv -y
python3 -m venv .venv
source ./.venv/bin/activate

# nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .

# Diffusion-Planner
cd ~/work/Diffusion-Planner/
python -m pip install pip==24.1

pip install -r requirements_nuplan-devkit_fixed.txt
pip install -r requirements.txt
pip install -e .

# check torch
python3 -c "import torch; print(torch.cuda.is_available())"

# (Optional) ros
./util_scripts/download_ros-hubmle.sh
```

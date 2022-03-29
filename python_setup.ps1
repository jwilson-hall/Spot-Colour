# take command line args

param (
    [Parameter()]
    # path to the file taken as a arg
    [string]$req_args = '.\requirements.txt'
)

Write-Output "Installing new virtual env"

# get virtual env
pip install virtualenv

# create new virtual to run project
python -m venv fyp_env
Write-Output "activate the new virt env"

# activate the new env
. .\fyp_env\Scripts\Activate

# install pipenv
Write-Output "installing pkgs......."
pip install -r $req_args
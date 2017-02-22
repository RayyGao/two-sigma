# The following is taken from:
# https://svds.com/jupyter-notebook-best-practices-for-data-science/

# Jupyter 4.x is now out! If you use the incredibly useful Anaconda
# distribution, you can upgrade with conda update jupyter. More options
# covered here.

# Further, config files are stored in ~/.jupyter by default. This directory is
# the JUPYTER_CONFIG_DIR environment variable. I’m going to first describe the
# above with the default config, and go into the complicated way of doing with
# Jupyter’s version of profiles.

# The default config file is found at: ~/.jupyter/jupyter_notebook_config.py

# If you don’t have this file, run: jupyter notebook --generate-config to
# create this file.

# Add the following text to the top of the file:

_c = get_config()
# If you want to auto-save .html and .py versions of your notebook:
# modified from: https://github.com/ipython/ipython/issues/8009
import os
from subprocess import check_call


def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return  # only do this for notebooks
    d, fname = os.path.split(os_path)
    check_call(['jupyter', 'nbconvert', '--to', 'script', fname], cwd=d)
    check_call(['jupyter', 'nbconvert', '--to', 'html', fname], cwd=d)

_c.FileContentsManager.post_save_hook = post_save

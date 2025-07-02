# tutorial-query-nomad-archive

## Development

Cloning the source code:
```
git clone git@gitlab.mpcdf.mpg.de:nomad-lab/ai-toolkit/tutorial-query-nomad-archive.git
cd tutorial-query-nomad-archive
git checkout refactoring # use a special branch
```

Creating a local environment fo the Python dependencies:
```bash
pyenv local 3.9
python -m venv .venv
source ./.venv/bin/activate
pip install pip-tools
pip-compile  --extra-index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple requirements.in
```


Running notebook image and mounting local folder into teh work directory:
```
docker run --rm -it -e DOCKER_STACKS_JUPYTER_CMD=notebook -p 8888:8888 -v $PWD:/home/jovyan/work gitlab-registry.mpcdf.mpg.de/nomad-lab/ai-toolkit/tutorial-query-nomad-archive:refactoring
```

Building the image (advanced):
```
docker build --pull --rm -f "Dockerfile" -t gitlab-registry.mpcdf.mpg.de/nomad-lab/ai-toolkit/tutorial-query-nomad-archive:refactoring "."
```

Running container with sudo feature (advanced):
```
docker run --rm -it --user root -e GRANT_SUDO=yes -e DOCKER_STACKS_JUPYTER_CMD=notebook -p 8888:8888 -v $PWD:/home/jovyan/work gitlab-registry.mpcdf.mpg.de/nomad-lab/ai-toolkit/tutorial-query-nomad-archive:refactoring
```


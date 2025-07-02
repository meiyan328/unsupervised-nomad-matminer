# tutorial-convolutional-nn

## Known issues
- kerase-vis is not maintained anymore (since Apr 20, 2020)
  - only works with: numpy==1.22.4, scipy==1.2.3

## Development

Cloning the source code:
```
git clone git@gitlab.mpcdf.mpg.de:nomad-lab/ai-toolkit/tutorial-convolutional-nn.git
cd tutorial-convolutional-nn
git checkout updates # use a special branch
```

Running notebook image and mounting local folder into teh work directory:
```
docker run --rm -it -e DOCKER_STACKS_JUPYTER_CMD=notebook -p 8888:8888 -v $PWD:/home/jovyan/work gitlab-registry.mpcdf.mpg.de/nomad-lab/ai-toolkit/tutorial-convolutional-nn:updates
```

Building the image (advanced):
```
docker build --pull --rm -f "Dockerfile" -t gitlab-registry.mpcdf.mpg.de/nomad-lab/ai-toolkit/tutorial-convolutional-nn:updates "."
```

Running container with sudo feature (advanced):
```
docker run --rm -it --user root -e GRANT_SUDO=yes -e DOCKER_STACKS_JUPYTER_CMD=notebook -p 8888:8888 -v $PWD:/home/jovyan/work gitlab-registry.mpcdf.mpg.de/nomad-lab/ai-toolkit/tutorial-convolutional-nn:updates
```

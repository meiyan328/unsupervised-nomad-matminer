ARG BUILDER_BASE_IMAGE=jupyter/base-notebook:python-3.7.12
FROM $BUILDER_BASE_IMAGE


# ================================================================================
# Linux applications and libraries
# ================================================================================

USER root

RUN apt update --yes \
 && apt install --yes --quiet --no-install-recommends \
    git \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

# RUN apt update --yes \
#  && apt install --yes --quiet --no-install-recommends \
#     build-essential \
#     cmake \
#     git \
#  && apt clean \
#  && rm -rf /var/lib/apt/lists/*


# ================================================================================
# Python environment
# ================================================================================

USER ${NB_UID}

# RUN mamba install --yes \
#         'numpy' \
#         'pandas' \
#         'hdbscan' \
#         'scikit-learn' \
#         'plotly' \
#         'bokeh' \
#         'matplotlib' \
#         'ipywidgets' \
#  && mamba clean --all -f -y \
#  && fix-permissions "${CONDA_DIR}" \
#  && fix-permissions "/home/${NB_USER}"


# RUN pip install --no-cache-dir --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple \
#         'nomad-lab' \
#         'numpy' \
#         'pandas' \
#         'hdbscan' \
#         'scikit-learn' \
#         'plotly' \
#         'bokeh' \
#         'matplotlib' \
#         'ipywidgets' \
#         'jupyter_jsmol==2021.3.0' \
#  && fix-permissions "${CONDA_DIR}" \
#  && fix-permissions "/home/${NB_USER}"

# Install from the requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.in /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.in \
 && fix-permissions "${CONDA_DIR}" \
 && fix-permissions "/home/${NB_USER}"

# ================================================================================
# Switch back to jovyan to avoid accidental container runs as root
# ================================================================================

USER ${NB_UID}
WORKDIR ${HOME}

COPY --chown=${NB_UID}:${NB_GID} data/ data/
COPY --chown=${NB_UID}:${NB_GID} assets/ assets/
COPY --chown=${NB_UID}:${NB_GID} convolutional_nn.ipynb .

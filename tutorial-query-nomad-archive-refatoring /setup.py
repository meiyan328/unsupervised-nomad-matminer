import json
from setuptools import setup, find_packages

with open('metainfo.json') as file:
    metainfo = json.load(file)

setup(
    name='query_nomad_archive',
    version='1.0',
    author=', '.join(metainfo['authors']),
    author_email=metainfo['email'],
    url=metainfo['url'],
    description=metainfo['title'],
    long_description=metainfo['description'],
    packages=find_packages(),
    install_requires=['nomad-lab', 'numpy', 'pandas', 'matplotlib','scikit-learn', 'plotly', 'mendeleev', 'tqdm', 'pydpc', 'seaborn', 'hdbscan', 'jupyter_jsmol==2021.3.0','bokeh','ipywidgets','typing'],
)

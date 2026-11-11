#!/usr/bin/env bash
# Jupyter for the organisers' timing run of /code/predict_notebook.ipynb (port 9777, password and token zac2026).
cd /code
jupyter lab --port 9777 --ip 0.0.0.0 --NotebookApp.password='zac2026' --NotebookApp.token='zac2026' --ServerApp.token='zac2026' --allow-root --no-browser

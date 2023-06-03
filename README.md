# hitl_chem

## Human-in-the-loop for compoud highligt color optimization

## requirements
 - rdkit
 - optuna
 - optuna-dashboard

## how to use


 ```python
 # terminal 1
 $ python main.py
 # terminal 2
 $ optuna-dashboard sqlite:///db.sqlite3 --artifact-dir ./artifact/
 ```
 go to http://localhost:8080

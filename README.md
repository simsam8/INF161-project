# INF161-project

python versjon brukt: 3.11

## Eksterne pakker

- matplotlib
- scikit-learn
- pandas 
- numpy
- seaborn
- plotly
- flask
- waitress

## Oppsett

1. Lag et python miljø og aktiver det
2. Installer pakker med: pip install -r requirements.txt
3. raw_data mappen fra mittuib må ligge i root.
4. Kjør INF161project.py for modellutvalg og trene beste modellering
5. Kjør app.py for nettsiden. Adressen er på localhost:8080



## Filer

I mappen classes finner du alle python klasser for data prossesering, analyse, feature engineering og modellering:

- DataCleaning.py
- DataExploration.py
- ModelEvaluation.py
- FeatureEngineering.py


I root mappen finner du:

- app.py:                   Fil for å kjøre nettside
- DataExploration.ipynb:    Jupyter notebook for dataanalyse
- INF161project.py:         Fil for modelutvalg, trening og prediksjon
- predictions.csv:          Prediksjon for 2023 data
- program_log:              Resultat fra modelutvalg
- rapport.md:               Markdown fil til å genere rapport
- requirements:             Liste over pakker brukt i prosjektet
- models/model.pkl:         Ferdigtrent modell
- templates/index.html:     Html for nettside
- images/:                  Bildemappe for rapport

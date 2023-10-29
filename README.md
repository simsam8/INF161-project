# INF161-project

## Oppsett

python versjon brukt: 3.11

1. Lag et python miljø og aktiver det
2. Installer pakker med: pip install -r requirements.txt
3. Kjør INF161project.py for modellutvalg og trene beste modellering
4. Kjør app.py for nettsiden. Adressen er på localhost:8080



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

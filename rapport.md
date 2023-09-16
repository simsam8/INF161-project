# Rapport

## Vasking av data

Vasking av data er hådtert av klassen DataCleaning.
Den har 4 innebygde metoder:

- clean_traffic_data
- clean_weather_data
- create_dataset
- get_dataset

### clean_traffic_data
Metoden clean_traffic_data leser av rå trafikkdata, vasker dataen og returner et datasett til videre prossesering.
Jeg valgte å fjerne de fleste av kolonnene, ettersom vi bare er interresert i total trafikkmende.
Kolonnene jeg fjernet besto av enten bare tomme verdier, eller en eller to unike verdier som ikke er relevant
for oppgaven.
-- Legg inn bilde av beskrivelse av datasettet --

Noen rader hadde samme tidspunkt, på grunn av tilbakestilling av klokken hvert år i oktober.
Jeg valge å fjerne radene som representerer den tilbakestilte tiden.
Det er snakk om 8 rader som blir fjernet, og som ikke har stor påvirkning på selve datasettet.
For å slippe å ta hensyn til tidsoner, valgte jeg bare å fjerne de.

-- Legg inn bilde av disse radene + forskjell med og uten de -- 


### clean_weather_data
Metoden clean_weather_data leser av alle filene om værdata i Florida, og slår dem sammen til et datasett.
Alle verdier som var 9999.9 ble gjort om til Nan verdier.
Datasettet er blitt resamplet til 1 times intervaller for å matche trafikkdatasettet.

I kolonnen med globalstråling har jeg satt alle negative verdier til 0.
Realistiske målinger av stråling går ikke under 0, og i henhold til datasettet,
har det nesten ingen påvirkning når de endres.

-- Legg inn bilde av forskjell --

Kolonnen med relativ luftfuktighet valgte jeg å fjerne.
Mesteparten av radene har manglende verdier, og vil dermed ha lite nytte sammen med de andre
kolonnene.

-- Legg inn bilde av antall rader med og uten verdier --

### create_dataset
Denne metoden kombinerer vasket trafikkdata og værdata til et datasett.
Den kombinerer datasettene på indeksen deres, som er indeksert på tidspunkt.

### get_dataset
Denne metoden kombinerer alle de forrige metodene.
Det er bare nødvendig å kalle på denne metoden for å få det ferdigvaskede datasettet.

## Data analyse

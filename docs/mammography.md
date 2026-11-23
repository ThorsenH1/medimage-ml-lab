# CBIS-DDSM mammografi

Oppgaven graderer malignitetsrisiko per bryst i et datasett med allerede annoterte lesjoner. Hver enhet kan ha CC-projeksjon, MLO-projeksjon eller begge. Vevsregionen segmenteres med en enkel forgrunnsmetode, og helbildet orienteres med brystveggen mot venstre. Datasettet beholder projeksjonsmasken under trening og inferens.

## Eksperiment

Den globale referansen bruker én nedskalert representasjon per tilgjengelig projeksjon. Oppmerksomhetsvarianten ser helbildet og seks overlappende utsnitt per projeksjon. Gated attention kombinerer utsnittene, og oppmerksomhet på tvers av projeksjonene kombinerer CC og MLO. Klasseetiketten er malign når minst én av de registrerte lesjonene på brystet er malign. Hovedmodellene bruker helbilder som inngang.

Pasienter deles på tvers av begge lesjonstyper. 31 pasienter med inkonsistent offisiell inndeling og 27 bryster med tvetydig kobling til mer enn ett bilde i samme projeksjon er ekskludert. For 282 radreferanser i forkalkningstestsettet ble innebygde bilde-ID-er og helbildedimensjoner brukt til å gjenopprette manglende metadata. Dette økte testsettet fra 221 til 336 bryster. Trenings- og valideringsbilder og tilhørende rader var byteidentiske gjennom rettingen.

Den endelige kohorten består av 1 039 bryster til trening, 260 til validering og 336 til test. Modellen ble valgt etter validerings-AUROC. Høyere dimensjon og mer oppmerksomhet kan føre til overtilpasning, spesielt med rundt tusen treningsenheter.

## Begrensninger og utdanningsverdi

Dataene er et retrospektivt, lesjonsanriket utvalg av digitalisert film. Utfallet, prevalensen og den tilgjengelige bildekvaliteten avgrenser hva målt ytelse betyr. Enkelte bryster har bare én projeksjon. Undergruppeanalysen viser resultatene for masse, forkalkning og projeksjonsdekning. [Datasett og dataartikkel hos TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

Prosjektet viser hvordan valg av målvariabel, koblingen mellom DICOM-metadata og JPEG og validering på pasientnivå kan endre et forskningsforsøk.

## Inferens

Bruk helbilder av samme bryst. Utelat `--cc` eller `--mlo` hvis bare én projeksjon finnes.

```powershell
& $p -m medimage.radiography_predict --checkpoint artifacts/runs/mammo-global/best.pt `
  --calibration artifacts/evaluation/mammo/calibration.json --cc 'C:\bilder\cc.jpg' --mlo 'C:\bilder\mlo.jpg' `
  --output artifacts/inference/mammo.json
```

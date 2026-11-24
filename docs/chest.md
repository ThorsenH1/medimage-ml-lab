# NIH ChestX-ray14

Oppgaven gir 14 binære utganger for hvert frontale thoraxbilde. Modellen med etikettspesifikke spørringer leser lokale trekk på et 5 × 5-gitter, modellerer samspill mellom etiketter og legger stedsinformasjonen til den globale utgangen. Den globale referansen bruker samme ResNet-18-stamme. Oppskriftene har ulike epokebudsjetter og OneCycle-planer, så resultatforskjellen gjelder de fullstendige oppskriftene.

## Kohort og validering

Alle 112 120 bilder i de offisielle listene ble identifisert. Fem grupper av identiske originalbilder forekom på tvers av den første inndelingen, med ulike pasient-ID-er. Den endelige inndelingen grupperer slike aliaser sammen. Fire utviklingspasienter med til sammen 29 bilder ble ekskludert fordi gruppene deres forekom i det offisielle testsettet. Testlista ble beholdt i sin helhet.

Treningsdelen består av 76 084 bilder fra 24 643 pasient-ID-er. En iterativ multilabel-splitt på grupper gir 10 411 valideringsbilder fra 3 361 andre pasient-ID-er. Offisielt testsett består av 25 596 bilder fra 2 797 pasient-ID-er, organisert i 2 795 grupper etter koblinger fra identiske bilder. Valideringen har 22 bilder med hernie, så usikkerheten for den sjeldne etiketten er høy.

Epoken og modellen velges etter gjennomsnittet av makro-AUROC og makro-gjennomsnittlig presisjon på valideringssettet. Hver etikett kalibreres separat med logistisk skalering. Rapporten måler rangering, kalibrering, resultater for undergrupper og pasientklustrede bootstrap-intervaller. Valideringsbaserte F1-grenser brukes bare til å beskrive testytelse.

## Begrensninger og utdanningsverdi

Etikettene kommer fra journalrapporter. Den innlærte rangeringen kan reflektere opptaksprojeksjon, behandlingsutstyr og bildemønstre som samvarierer med funnet. AP-/PA-, alders- og kjønnsanalysene beskriver observerte forskjeller i datasettet. Ekstern sykehusvalidering, klinisk terskelvalg og prospektiv evaluering krever egne studier. [NIH-kilden](https://nihcc.app.box.com/v/ChestXray-NIHCC) · [RSNA-datakort](https://atlas.rsna.org/cards/66962171-1dcb-431d-89b0-08a561f1c8ed)

Prosjektet viser hvorfor presisjon–recall, prevalens og kalibrering må leses sammen og hvordan gjentatte bilder per pasient påvirker usikkerheten.

## Inferens

Trenings- og evalueringsskriptet skriver `evaluation-plan.json` med det valgte modellnavnet. Bruk dette navnet i checkpointbanen.

```powershell
& $p -m medimage.radiography_predict --checkpoint artifacts/runs/chest-global/best.pt `
  --calibration artifacts/evaluation/chest/calibration.json --image 'C:\bilder\frontalbilde.png' `
  --output artifacts/inference/chest.json
```

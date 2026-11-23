# RSNA LumbarDISC MR korsrygg

Modellen beregner tre ordnede grader for 25 kombinasjoner av funn og nivå. De tre bildeseriene er sagittal T2/STIR, sagittal T1 og aksial T2. Pipeline leser pikselavstand, bildeposisjon og orientering og konverterer resamplede koordinater tilbake til pasientrom i millimeter.

## Lokalisering og klassifisering

1. Sentralsagittale bildebeskrivelser brukes i en kjernemodell for fem landemerkepunkter. RBF-regresjonen kombinerer to båndbredder og bruker en median fra sine treningsfolder som reserve.
2. Avstander mellom landemerkene og målposisjonene estimeres på treningsfoldene. Det resulterende atlaset overfører landemerker til venstre og høyre foramina og subartikulære områder.
3. Målpunktene projiseres på aktuelle serier i millimeter. Hvert 2,5D-utsnitt består av tre bildesnitt. Manglende serier gir et eksplisitt fraværsignal.
4. En delt ResNet-18 koder utsnittene. Oppmerksomheten kombinerer nivå, funntype, bildekontekst og lokaliseringskvalitet. Tapet vekter alvorlige grader og modellerer ordinal avstand og alvorlig stenose i studiet som helhet.

## Kontroll mot datalekkasje

De historiske fem foldene på 1 975 undersøkelser brukes uendret. For hver ytre fold bruker lokaliseringen av valideringsbildene de fire andre foldene. Lokaliseringen av et treningsbilde holder ute både bildets egen fold og den ytre valideringsfolden. De 15 forskjellige fittene lagrer opphavet til hver ROI. Dermed bruker også klassifikatorens treningsbilder estimerte landemerker. Ytre valideringsetiketter inngår i rapportering, mens sluttvektene følger et fast treningsforløp og gjennomsnittet av de siste tre epokene.

Tilgjengelige valideringsannotasjoner brukes i ett eksplisitt følsomhetsforsøk på fold 0. Det måler hvor mye samme modell endrer seg når koordinatene er kjent. Det oppgis separat i resultatene.

## Evaluering og inferens

Referansen er en historisk likt vektet femfoldsmodell med globale 2,5D-bilder. Begge modeller vurderes på de samme foldene etter den eksakte RSNA-log-loss-metrikken. Klassifikatoren og en på forhånd fastlagt 50/50-blanding må slå referansen på minst tre folder og ha et parvis bootstrap-intervall med øvre grense under null for å få erstatte den. En utforskende temperaturkalibrering rapporteres som egen analyse.

Det finnes én lokal testundersøkelse. Inferens må produsere 25 unike, endelige rader i riktig rekkefølge med sannsynligheter som summerer til én. Den undersøkelsen er et format- og integrasjonstestsett. [Offisiell utfordring og data](https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge)

```powershell
& $p -m medimage.spine_infer --root ../lumbar_spine_degenerative_classification `
  --cache artifacts/cache/spine-test --assessment artifacts/evaluation/spine/metrics.json `
  --localization artifacts/spine/localization --runs artifacts/spine/runs `
  --output artifacts/inference/spine.csv
```

Ved evaluering med en historisk referansemodell, bruk `--baseline-oof` i `medimage.spine_evaluate`. Hvis den valgte konfigurasjonen bruker globale sannsynligheter, oppgi dens validerte CSV med `--baseline-submission`.

`medimage.spine_reference` gjenskaper referansens inferens fra rå DICOM for fem lokale checkpoint. Referansen bruker 224 piksler, fem lineært fordelte snitt per serie, en delt ResNet-18 og gjennomsnittspooling. De historiske checkpointene ble trent fra ImageNet-initialisering over de samme foldene med AdamW og FP16.

## Begrensninger og utdanningsverdi

Sekvensvalg ved flere serier favoriserer dekningsområde for aksial T2 og antall snitt for de sagittale seriene. Den statistiske atlasmodellen kan feile ved uvanlig anatomi, stort avvik i orientering eller delvis bildefelt. Institusjonsuavhengig og klinisk ytelse må etableres i separate studier. Prosjektet viser hvordan fysisk bildegeometri, nestet kryssvalidering og en målspesifikk metrikk virker sammen.

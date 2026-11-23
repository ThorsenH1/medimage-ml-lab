# Faglig gjennomgang

Bruk [resultatrapporten](results.md) og [forskningskildene](research.md) som utgangspunkt for å presentere prosjektene.

| Tema | Eksempel i prosjektet | Spørsmål å drøfte |
| --- | --- | --- |
| Bildefysikk | MR-utsnitt styres av fysisk bredde i millimeter og en lokal basis for hver DICOM-skive | Hvordan endrer anisotrop pikselavstand et kvadratisk utsnitt definert i pikselkoordinater? |
| Representativt utvalg | CBIS-DDSM inkluderer bryster som allerede har annoterte lesjoner | Hva skjer med presisjon og kalibrering når prevalensen endres i en screeningpopulasjon? |
| Datalekkasje | Samme pasient kan ha flere bilder. En separat lokaliseringsmodell kan også lære den samme pasientens koordinater | Hvilke nivåer må holdes ute når et trent system leverer input til neste modell? |
| Ubalanse | Thoraxprosjektet rapporterer AUROC, gjennomsnittlig presisjon og prevalens for alle 14 etiketter | Hvilke etiketter rangeres godt samtidig som den absolutte presisjonen er lav? |
| Modellvalg | Global pooling og oppmerksomhet sammenlignes på valideringssettet før test | Hvilke bevis kreves for at mer kompleks arkitektur skal overta? |
| Usikkerhet | Bootstrapping trekker hele pasienter og sammenligner MR-modellene med de samme studiene | Hvorfor kan intervallene bli misvisende hvis man trekker enkeltbilder? |

## Reproduserbar analyse

1. Undersøk dataauditen og forklar hvordan kohorten ble satt sammen. Vis hvordan en pasientidentifikator påvirker inndelingen uten å ta med pasientnivårader i presentasjonen.
2. Tegn tre trinn fra råbilde til sannsynlighet for hvert prosjekt. Forklar hvilke parametere som ble lært fra data og hvilke regler som var fastlagt.
3. Velg ett resultat som utfordrer den opprinnelige hypotesen. Vis valideringsmålet, hvilke alternativer det sammenligner og hvordan valget ble låst før test.
4. Diskuter hva et lokalt testresultat sier om målpopulasjonen, og hvilke data som mangler for å måle overførbarhet til andre institusjoner.

# MedImage ML Lab

Tre maskinlæringsprosjekter for medisinske bilder. Hvert prosjekt dekker hele arbeidsflyten fra rådata og datasplitt til GPU-trening, evaluering og lokal inferens. Utviklet for forskning og utdanning.

| Prosjekt | Oppgave | Eksperiment |
| --- | --- | --- |
| [NIH ChestX-ray14](docs/chest.md) | 14 uavhengige funnetiketter per frontalbilde | ResNet-18 med global pooling sammenlignet med etikettspesifikk oppmerksomhet og samspill mellom etiketter |
| [CBIS-DDSM](docs/mammography.md) | Malignitetsklassifisering per bryst i et lesjonsanriket datasett | Global modell sammenlignet med oppmerksomhet over flere utsnitt og CC-/MLO-projeksjoner |
| [RSNA LumbarDISC](docs/lumbar.md) | Tre grader for fem tilstander ved fem nivåer i korsryggen | Nestet kryssvalidering av landemerker, DICOM-projeksjon i millimeter og 2,5D-klassifisering |

[Resultater og figurer](docs/results.md) · [Kilder og metodegrunnlag](docs/research.md) · [Faglig gjennomgang](docs/study-guide.md)

## Lokalt oppsett

Prosjektet er kjørt på Windows med Python 3.11, PyTorch 2.14.0 og CUDA 12.6 på en NVIDIA GeForce RTX 2080 SUPER med 8 GiB VRAM. Treningen bruker FP16 automatisk blandet presisjon og FP32 tapsberegning. Bildene lagres i filbaserte `uint8`-cacher som leses ved behov.

Med Python 3.11 tilgjengelig, kjør følgende fra rotmappen. CUDA-hjulene kommer fra PyTorch sitt eget pakkeregister. [PyTorch AMP-dokumentasjon](https://docs.pytorch.org/docs/2.14/amp.html)

```powershell
python -m venv .venv
$p = '.\.venv\Scripts\python.exe'
& $p -m pip install -r requirements.txt
& $p -m pip install torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
& $p -m pytest -q
```

Treningen bruker det offisielle ImageNet-1K-checkpointet for ResNet-18. Legg `resnet18-f37072fd.pth` i `hub/checkpoints` under mappen du gir som `--weights`. Treningsprogrammet verifiserer SHA-256-prefikset og bruker den lokale filen. [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) · [ResNet-18-vekter](https://download.pytorch.org/models/resnet18-f37072fd.pth)

Hent de tre datasettene fra kildene i prosjektbeskrivelsene. Oppgi rotmapper og vektmappe selv. Denne PC-en kan bruke Python-miljøet og de forhåndstrente ImageNet-vektene som allerede er installert.

```powershell
$p = 'D:\rsna2024_work\envs\rsna\python.exe'
$weights = 'D:\rsna2024_work\torch-cache'
pwsh -File scripts/Run-Radiography.ps1 -Python $p -Weights $weights -Task chest -DataRoot ../chest_xray
pwsh -File scripts/Run-Radiography.ps1 -Python $p -Weights $weights -Task mammo -DataRoot '../CBIS DDSM Breast Cancer Image Dataset'
pwsh -File scripts/Run-Spine.ps1 -Python $p -Weights $weights -DataRoot ../lumbar_spine_degenerative_classification
```

Kjør GPU-arbeidsflytene etter tur. Lokal `artifacts/` bør ha minst 8 GiB ledig plass til cacher, checkpoint og resultater. Samme kommando kan gjenoppta en avbrutt cache eller trening fra siste lagrede epoke. Python-modulene har `--help` for separate faser.

## Lokal inferens og prosjektfiler

- `medimage.radiography_predict` tar ett frontalbilde eller ett til to mammografibilder og returnerer kalibrerte forskningssannsynligheter.
- `medimage.spine_infer` leser rå DICOM, lokaliserer fem nivåer, beregner sannsynlighetene og validerer radformatet for RSNA.
- `artifacts/cache/` inneholder lokale kohorter og avledede bilder. Treningshistorikk, checkpoint, pasientnivåprediksjoner og evalueringsrapporter ligger også under `artifacts/`.
- `tests/` dekker blant annet pasientseparasjon, DICOM-geometri, manglende projeksjoner, eksakt metrikksamsvar, alvorlighetsmaskering og GPU-regning.

Repoet deler kildekode, metode og aggregerte tall. Pasientnivåfiler, datasett og modellvekter oppbevares lokalt. Modellene er forskningsprototyper uten klinisk validering.

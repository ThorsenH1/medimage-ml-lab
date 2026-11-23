# Kilder og metodegrunnlag

## Røntgen thorax

NIH sin opprinnelige artikkel beskriver rapportbasert etikettering for den første utgaven av datasettet. Arbeidet her bruker den senere utgaven med 14 funnetiketter og de lokalt leverte offisielle listene. Rapportbaserte etiketter kan være ufullstendige og avvike fra en uavhengig bildegranskning. [NIH-artikkel](https://arxiv.org/abs/1705.02315) · [RSNA-datakort](https://atlas.rsna.org/cards/66962171-1dcb-431d-89b0-08a561f1c8ed)

## Mammografi

CBIS-DDSM gir patologisvar, fullmammogrammer og separate lesjonsannotasjoner. En oppgave som gjelder malignitet blant allerede annoterte bryster har et annet datagrunnlag enn generell screening. [TCIA-samlingen](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) · [Dataartikkelen](https://doi.org/10.1038/sdata.2017.177)

Den lokale JPEG-distribusjonen knytter sammen `image file path` fra kasusbeskrivelsen og `image_path` fra DICOM-indeksen. En tom beskrivelse i indeksfilen håndteres med eksakt serie- og studie-ID fra helbildereferansen og kontroll av full bildestørrelse. [Den lokale JPEG-distribusjonens kildeside](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

## MR korsrygg

LumbarDISC ble samlet for gradering av degenerative funn ved fem nivåer. Konkurransemetrikken bruker vektet log loss for tre familier av funn og en egen komponent for alvorlig spinalkanalstenose. [RSNA-dataartikkelen](https://doi.org/10.1148/ryai.250480) · [Kaggle-konkurransen](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) · [Offisiell metrikkimplementasjon](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/521786)

Offentlige løsninger motiverte lokaliser-først-tilnærmingen. Denne implementasjonen bruker HOG og kjerneregresjon for nøkkelpunkter, DICOM-geometri for serieoverføring, en treningsfoldspesifikk anatomisk atlasforskyvning og en delt 2,5D-klassifikator. [Vinnerens løsningsbeskrivelse](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution) · [Andreplassens kode](https://github.com/brendanartley/RSNA-2024-Competition) · [Fjerdeplassens kode](https://github.com/yu4u/kaggle-rsna2024-4th) · [Sjuendeplassens totrinnsløsning](https://github.com/lhwcv/solution-rsna-2024-lumbar-spine) · [Niendeplassens kode](https://github.com/adamnarai/kaggle-rsna-2024)

Tredjeplassen overførte sagittale koordinater til skrå aksialsnitt i fysisk rom. Femteplassen modellerte relasjoner mellom nivåer etter varmekartbasert lokalisering. Begge fremgangsmåtene motiverer geometri og nivårelasjoner som kontrollerbare komponenter i dette forsøket. [Tredjeplassens analyse](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453) · [Femteplassens løsning](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution)

Fysisk projeksjon bruker `ImagePositionPatient`, `ImageOrientationPatient` og `PixelSpacing` i DICOM-koordinater. OpenCVs pikselmidtpunkt-konvensjon er innarbeidet i letterbox-transformasjonen. [DICOM-standarden for bildeposisjon og orientering](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)

## Trening på arbeidsstasjonen

Tre eksperimenter bruker ImageNet-initialisert ResNet-18. CUDA-AMP bruker FP16 autocast med dynamisk gradientskalering. Læringsrater, manifest-hasher og modellvalg lagres i hvert forsøk. [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) · [PyTorch AMP-oppskrift](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

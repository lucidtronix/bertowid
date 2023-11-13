# BERTowid
This repo contains the code for the paper ["Trajectories of Sentiment in 11,816 Psychoactive Narrativesoactive Experiences"](https://www.biorxiv.org/content/10.1101/2022.06.02.494544)

<img width="1512" alt="figure1" src="https://github.com/lucidtronix/bertowid/assets/2604962/b17bac6e-d079-414d-be2e-e900ea1de62c">

This figure represents an overview of the study showing data sources, models and a selection of key results. Data from Wikipedia, IMDB, Rotten Tomatoes, Reddit, Erowid, and the PDSP  was used in pre-training and training the 3 models: BERTowid, BERTiment, and CCA. A) BERTiment’s concordance with a clinical-psychiatrist emotion adjudication in Erowid testimonials. B) IMDB movie review hedonic-tone classifier correlation with the 28 emotions annotated in Reddit.  In C)-G) we show BERTowid output heads evaluated on held-out test set testimonials: C) BERTowid psychoactive classification, mean precision per class. D) Self-reported gender classification ROC curve. E) Predict age from testimonials F) Predict CCA factors per-testimonial weighting for each of the 11 axes of the receptor-experience space. G) Regress receptor affinity directly from testimonial text.  Tiling inferences from BERT models along the narrative of the testimonials we construct trajectories, for example, H) shows BERTiment’s predicted emotion: Love for select drugs over the course of a trip (note the prominence of MDMA).  I) BERTowid’s predicted trajectories for the semantic tag: Mystical Experience (note the prominence of DMT).

<img width="1512" alt="figure2" src="https://github.com/lucidtronix/bertowid/assets/2604962/c2c951f3-2a82-4c08-8a9e-e18cab239c00">
<img width="1512" alt="figure3" src="https://github.com/lucidtronix/bertowid/assets/2604962/bbf10d2a-73a9-41bd-a2d0-9a85bbe36805">
<img width="1511" alt="figure4" src="https://github.com/lucidtronix/bertowid/assets/2604962/3c61614b-fd04-4e7c-978d-3a618aeb5787">
<img width="1512" alt="figure5" src="https://github.com/lucidtronix/bertowid/assets/2604962/3967be7a-a904-4242-9701-3ff94ee35649">
<img width="1511" alt="figure6" src="https://github.com/lucidtronix/bertowid/assets/2604962/f8aa9857-8b16-40b7-9bef-9884edc01b86">

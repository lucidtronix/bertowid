# BERTowid
This repo contains the code for the paper ["Trajectories of Sentiment in 11,816 Psychoactive Narratives"](https://doi.org/10.1002/hup.2889)

<img width="1512" alt="figure1" src="https://github.com/lucidtronix/bertowid/assets/2604962/b17bac6e-d079-414d-be2e-e900ea1de62c">
This figure illustrates the three models with a selection of key results. Left: the dominant component, CCA 0, found by the self-supervised learning CCA method. One extreme of CCA 0 encodes concepts of somatic suffering displayed in blue, while the other pole encompasses visual beauty and is displayed in red.  All 52 drugs included in the study are shown in (a) with the ranking along CCA 0. (b) shows a brain surface map of CCA 0. (c) shows receptor clouds, note how the red visual/beauty pole is entirely serotonergic (5HT*) while the somatic/suffering pole in blue highlights several different neurotransmitter types including opioid (MOR, DOR), GABA and acetylcholine (M1, M4). (d) shows word clouds with font size determined by their CCA 0 weighting. (e) shows a surface view of CCA 0 mapping into the brain, note how the visual pole highlights the visual cortex. Middle: results from the transfer learning model, BERTowid, which is a multi-task classification and regressing transformer trained directly on Erowid testimonials.  All results are from test set testimonials, which were not used in training. (f) Pearson correlation with the 11 CCA factors per-testimonial weightings. (g) self-reported gender ROC curve. (h) Pearson correlation with self-reported age. (i) mean precision per psychoactive class.  Tiling inferences from BERT models along the narrative of the testimonials we construct trajectories, for clarity we only show a few of 52 drugs here.  (j) trajectories for the semantic tag of “Mystical Experiences”, note the prominence of DMT. Right: pretrained model results from BERTiment, trained on an entirely different text corpus to classify emotions. (k) BERTiment’s concordance with a clinical-psychiatrist emotion adjudication in Erowid testimonials. (l) IMDB movie review hedonic-tone classifier correlation with the 28 emotions inferred on Erowid. (m) BERTiment Sadness trajectories, note how the antidepressants track with each other and are initially quite elevated. (n) BERTiment Love trajectories, note the prominence of MDMA.

<img width="1512" alt="figure2" src="https://github.com/lucidtronix/bertowid/assets/2604962/c2c951f3-2a82-4c08-8a9e-e18cab239c00">
Figure 2: Trajectories and Drug Taxonomies
Left: BERTiment trajectories for the emotion “Admiration” at the pharmacological level (a), chemical level (b) and individual drug level (c), see Table 2 for the full drug taxonomy. Note that for clarity we have selected only 12 representative drugs of the 52 included, see other figures for comparisons involving all drugs. Shaded regions indicated +/- intra-drug standard deviation. Right: BERTowid trajectories for each of the 3 different levels of drug classification (from the left panel) on metadata tags “Mystical Experiences” (d), “Addiction Habituation” (e), “Depression” (f), and “Rave Dance Event” (g).  Note the concordance between the entactogens MDA and MDMA and the antidepressants sertraline and venlafaxine.

<img width="1512" alt="figure3" src="https://github.com/lucidtronix/bertowid/assets/2604962/bbf10d2a-73a9-41bd-a2d0-9a85bbe36805">
<img width="1511" alt="figure4" src="https://github.com/lucidtronix/bertowid/assets/2604962/3c61614b-fd04-4e7c-978d-3a618aeb5787">
<img width="1512" alt="figure5" src="https://github.com/lucidtronix/bertowid/assets/2604962/3967be7a-a904-4242-9701-3ff94ee35649">
<img width="1511" alt="figure6" src="https://github.com/lucidtronix/bertowid/assets/2604962/f8aa9857-8b16-40b7-9bef-9884edc01b86">

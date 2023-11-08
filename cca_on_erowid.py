# cca_on_erowid.py
import os
import csv
import re
from collections import Counter, defaultdict

import argparse
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sb
import wordcloud
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from scipy.stats import pearsonr

from defines import DRUG_NAMES, DRUGS_TO_USE, RECEPTORS_TO_USE

CHARS_TO_REMOVE = " .,~{}[]()!?@#$-:+;/_\"'"
MIN_LINE_CHARS = 10
MEDIAN_AGE = 21
COMMON_WORDS = ['the', 'this', 'that', 'not', 'and', 'have', 'there', 'all', 'then', 'what', 'but', 'would', 'for', 'with', 'will', 'was', 'thing',
                'get', 'could', 'from', 'more', 'etc', 'who', 'out', 'another', 'like', 'too', 'while', 'about', 'more', 'less', 'way', 'on',
                'she', 'her', 'him', 'his', 'our', "i'm", 'i’m', 'are', 'can’t', "i'd", 'i’d',  'ich', 'der', 'das',
                "didn't", "don't", "dont", "i've", "it's", "wasn’t", "can't", "wouldn't", "couldn't", "couldn´t", "won't", "i'll",
                'them', 'were', 'they', 'through', 'back', 'being', 'only', 'also',
                'went', 'some', 'again', 'into', 'after', 'around', 'down', 'just', 'very', 'things', 'when', 'over', 'other', 'before',
                'because', 'which', 'much', 'took', 'than', 'before', 'still', 'didn’t',
                'it’s', 'i’ve', 'didnt', 'didn´t', 'couldnt', 'couldn’t', 'their',
                'don’t', "that's", 'won’t', 'und', 'che', 'que',
                'μg/kg', 'mgs', "mg's", 'hcl', 'indole',
                'plant', 'material', 'powder',  'crystal', 'crystals', 'tar', 'resin', 'pill','ground', 'crushed', 'blotter', 'tab',
                'pill', 'pills', 'pipe', 'smoke', 'smokes', 'smoked','toke', 'blotter', 'tab', 'tabs', 'line', 'lines', 'dose', 'doses', 'dosage', 'hit', 'hits', 'bowl',
                'trip', 'trips', 'tripping', 'tripped', 'trippy', 'k.hole', 'k-hole', 'khole',
                'roll', 'rolls', 'rolling', 'rolled','vaporized', 'vaporize',  'roll', 'rolling', 'rolled', 'nasal', 'bong', 'foil', 'root', 'bark',
                'syringe', 'needle',   'joint', 'joints', 'vial', 'bag', 'inject', 'drugs',
                'vials', 'caps', 'bottle', 'prescribe', 'prescribed', 'medication', 'medications',
                'insufflated', 'insufflate', 'pill', 'tablet', 'tablets', 'edible', 'sublingual', 'vials',
                'snort', 'snorts', 'snorting', 'snorted', 'smoked','smokes', 'smoking', 'smoke',
                'das', '1999', '1/2', 'ten', 'substance', 'load', 'cherek', '5:00', '2001', '300', 'you', 'josh',
                 '2000', '1/4', '7.5' 'w/o', 'b/c', '150', '0.5', '10x', '8oz',
                'die', 'nicht', 'mit', 'mir', 'darla', 'sich', 'mich', 'ist', 'ein', 'war', 'den',
                'noch', 'een', 'auch', 'dass', 'hatte', 'auf', 'von', 'meine', 'als', 'eine', 'jul', 'jan', 'jun', 'aug', 'sep', 'dec', 'feb', 'apr',
                'einen', 'alal', 'sie', 'het', 'dem', 'aus', 'mark', 'aber', 'nach', 'marijuana',
                'des', 'approx', 'wavy', 'john','joe', 'burnt', 'wie', 'chris',
                ]

SKIP = ['-PRON-', 'exp', 'gender']
SKIP += DRUG_NAMES
SKIP += COMMON_WORDS


def parse_args():
    print(f'{sorted(set(DRUG_NAMES))}')
    parser = argparse.ArgumentParser()

    parser.add_argument('--drug_folder', default='psychedelics_mdma',
                        help='Folder of text dumps of testimonials, one drug per file.')
    parser.add_argument('--limit', default=128, type=int,
                        help='Maximum number of Testimonials per drug to load.')
    parser.add_argument('--pca_components', default=1000, type=int,
                        help='Number of PCA components to keep in reduced word count matrix')
    parser.add_argument('--cca_components', default=5, type=int,
                        help='Number of CCA components to find in reduced word count matrix')
    parser.add_argument('--stratify', choices=['male', 'female', 'old', 'young', None],
                        help='Stratify by sex or age.')
    parser.add_argument('--receptor_file', default='psychedelics_mdma',
                        help='Folder of text dumps of testimonials, one drug per file.')
    parser.add_argument('--min_word_occurrences', default=7, type=int,
                        help='Minimum number of word occurrences for word to be included in word count matrix')
    parser.add_argument('--affinities', default='PDSP_KiDatabase_corrected_now.csv',
                        help='File with affinity values for drugs and receptors.')
    parser.add_argument('--normalize', default='-', choices=[ 'exponent', 'by_drug', 'by_receptor'],
                        help='How to normalize the receptor affinity matrix')
    parser.add_argument('--id', default='run_name',
                        help='Name to identify this pipeline run.')
    parser.add_argument('--seed', default=123456, type=int,
                        help='Seed the random number generator')
    return parser.parse_args()


def run():
    args = parse_args()
    np.random.seed(args.seed)
    ccas = []
    cca_names = []
    selections = []
    cca, selected = find_latent_space_of_consciousness(max_testimonials_per_drug=args.limit,
                                                       pca_components=args.pca_components,
                                                       cca_components=args.cca_components,
                                                       drug_folder=args.drug_folder,
                                                       affinities=args.affinities,
                                                       normalize=args.normalize,
                                                       id=args.id,
                                                       stratify=args.stratify,
                                                       min_word_occurrences=args.min_word_occurrences,
                                                       )
    ccas.append(cca)
    cca_names.append(f'limit:{args.limit}, PCA:{args.pca_components} Stratify:{args.stratify}')
    selections.append(selected)
    # plot_cca_cross_correlations2(ccas, cca_components, cca_names, pca_components, limit, selections)


def find_latent_space_of_consciousness(max_testimonials_per_drug, pca_components, cca_components, drug_folder,
                                       affinities, normalize, id, stratify=None, permutation_tests=21, min_word_occurrences=12):
    #affinity_map, receptors = make_affinity_map('NEW_AFFINITY_MATRIX_nomenclature.csv', normalize=normalize)

    #affinity_map, receptors = make_affinity_map('NEW_AFFINITY_MATRIX_nomenclature_3_2022.csv', normalize=normalize)
    affinity_map, smile_map = make_affinity_map_pdsp(affinities)
    word_count_matrix, affinities, word_columns, selected, drugs, testimonial_totals = make_corpus(drug_folder, affinity_map, max_testimonials_per_drug,
                                                                               stratify, min_word_occurrences)
    print(f'Found {len(drugs)} drugs: {drugs}')
    pca, tfidf_reduced = pca_on_word_matrix(word_count_matrix, pca_components)
    cca, word_train_r, word_test_r, receptor_train_r, receptor_test_r = fit_cca_and_transform(cca_components, tfidf_reduced,
                                                                                              tfidf_reduced, affinities, affinities)

    receptor_cca_loads = np.vstack((np.asarray(RECEPTORS_TO_USE), np.asarray(cca.y_loadings_.T)))
    filename = f"./tsvs/{id}_max_{max_testimonials_per_drug}_pca_{pca_components}_{stratify}_cca_{cca_components}_on_{drug_folder}.tsv"
    np.savetxt(filename, receptor_cca_loads, delimiter="\t", fmt="%s")

    columns = [f'cca_{i}' for i in range(cca_components)]
    testimonial_cca_loads = pd.DataFrame(data=np.asarray(cca.x_scores_), columns=columns)

    testimonial_weight_filename = f"./tsvs/testimonial_{id}_max_{max_testimonials_per_drug}_pca_{pca_components}_{stratify}_cca_{cca_components}_on_{drug_folder}.tsv"
    testimonial_cca_loads['drug_idx'] = [x[0] for x in selected]
    testimonial_cca_loads['testimonial_idx'] = [x[1]+1 for x in selected]
    testimonial_cca_loads.to_csv(testimonial_weight_filename, sep="\t", index=False)

    #heatmap_correlations(cca, word_count_matrix, testimonial_totals, drug_folder)
    #receptor_cca_weights_graphic(affinity_map, receptors, cca)
    #receptor_cca_drug_correlations(affinity_map, receptors, cca)
    drug_loadings = sum_drug_loadings2(cca, testimonial_totals)
    # pca_loadings = np.dot(cca.x_loadings_.T, pca.components_[:pca_components, :]).T
    # word_loadings = np.argsort(pca_loadings, axis=0)
    #plot_drugs(word_loadings, drug_loadings)
    analyze_components(cca, pca, pca_components, word_columns, RECEPTORS_TO_USE,
                       drugs, drug_folder, drug_loadings,
                       name=id, limit=max_testimonials_per_drug)
    if permutation_tests > 0:
        danilos_permutation_test(cca_components, tfidf_reduced, affinities, permutation_tests, 1234)
        # danilos_permutation_test(cca_components, tfidf_reduced, affinities, permutation_tests, 4321)
        # danilos_permutation_test(cca_components, affinities, tfidf_reduced, permutation_tests, 1234)
    word_train, word_test, receptor_train, receptor_test = shuffle_and_split_data(word_count_matrix, affinities)
    #cca, word_train_r, word_test_r, receptor_train_r, receptor_test_r = fit_cca_and_transform(components, word_train, word_test, receptor_train, receptor_test)
    #plot_cca_cross_correlations(components, word_train_r, word_test_r, receptor_train_r, receptor_test_r)

    print(f'Saved file at: {filename}')
    return cca, selected


def heatmap_correlations(cca, word_count_matrix, testimonial_totals, drug_folder, drug_prefix='./testimonials/'):
    corrs = np.zeros((4, cca.x_scores_.shape[-1]))
    unique_words = np.count_nonzero(word_count_matrix, axis=-1)
    print(unique_words.shape)
    for component in range(cca.x_scores_.shape[-1]):
        rho1 = pearsonr(unique_words, cca.x_scores_[:, component])[0]
        rho2 = pearsonr(unique_words, cca.y_scores_[:, component])[0]
        corrs[0, component] = rho1
        corrs[1, component] = rho2
        print(f'Word complexity Pearson at component {component} is {rho1} {rho2}')

    drug_properties = pd.read_csv('drug_properties.tsv', sep='\t', header=None)
    zipt = {k: v for k,v in zip(drug_properties[0], drug_properties[4])}
    durations = []
    for f in sorted(os.listdir(drug_prefix + drug_folder)):
        if not f.endswith('.txt'):
            continue
        drug = f.replace('.txt', '')
        drug_count = testimonial_totals[drug]
        duration = float(zipt[drug])  #float(zipt[drug.lower()])
        durations.extend([duration] * drug_count)
    for component in range(cca.x_scores_.shape[-1]):
        rho1 = pearsonr(durations, cca.x_scores_[:, component])[0]
        rho2 = pearsonr(durations, cca.y_scores_[:, component])[0]
        corrs[2, component] = rho1
        corrs[3, component] = rho2
        print(f'Duration Pearson at component {component} is {rho1} {rho2}')

    fig, ax = plt.subplots(figsize=(11, 9))
    # plot heatmap
    sb.heatmap(corrs, cmap=sb.diverging_palette(220, 20, as_cmap=True), square=True, center=0, vmin=-1.0, vmax=1.0,
           linewidth=0.3, cbar_kws={"shrink": .4})
    plt.title('Correlations between Mode and Meta Data')
    yticks_labels = [f'Language\nComplexity\nx        \nSemantics', 'Language\nComplexity\nx        \nReceptors', 'Duration\nx        \nSemantics', 'Duration\nx        \nReceptors']
    xticks_labels = (range(1, cca.x_scores_.shape[-1]+1 ))
    plt.xlabel('Mode #')
    plt.yticks(np.arange(4)+0.5 , labels=yticks_labels)
    plt.xticks(np.arange(cca.x_scores_.shape[-1]) + 0.5, labels=xticks_labels)
    figure_path='./heatmap_correlations.png'
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def make_corpus(drug_folder, affinity_map, limit=None, stratify=None, min_word_occurrences=12, drug_prefix = './testimonials/'):
    drugs = {}
    selected = []
    documents = {}
    affinities = []
    meta_data = {}
    testimonial_totals = {}
    offset_testimonials = 0
    for f in sorted(os.listdir(drug_prefix + drug_folder)):
        if not f.endswith('.txt'):
            continue
        with open(os.path.join(drug_prefix, drug_folder, f)) as text:
            docs, affinity, idxs, drug, offset, meta = split_clean(f, text.read(), affinity_map, limit=limit, stratify=stratify,
                                                               offset_words=len(documents), offset_testimonials=offset_testimonials)
            offset_testimonials += offset
            testimonial_totals[f.replace('.txt', '').lower()] = len(docs)
            drugs.update(drug)
            meta_data[f] = meta
            documents.update(docs)
            selected.extend(idxs)
            affinities.extend(affinity)
    plot_testimonial_histogram(testimonial_totals)
    plot_meta_data(meta_data)
    print(f'Got {len(documents)} total testimonials.')
    frequency = Counter()
    counts_per_document = defaultdict(Counter)
    total_counts_per_document = Counter()
    total_documents_per_word = Counter()
    texts = [[word for word in documents[d].split()] for d in documents]

    for i, text in enumerate(texts):
        for token in text:
            frequency[token] += 1
            counts_per_document[i][token] += 1
            total_counts_per_document[i] += 1
    word_columns = [token for token in frequency if frequency[token] > min_word_occurrences]
    print(f'Total words:{len(frequency)} with frequency > {min_word_occurrences} total:{len(word_columns)} \nMost common 30:{frequency.most_common(30)}')
    word_count_matrix = np.zeros((len(documents), len(word_columns)))
    for i, d in enumerate(documents):
        for j, word in enumerate(word_columns):
            if word in counts_per_document[i]:
                word_count_matrix[i, j] = counts_per_document[i][word]
                total_documents_per_word[word] += 1
    affinities = np.array(affinities)
    print(f'Words count matrix shape:{word_count_matrix.shape} receptor affinities shape:{affinities.shape}. Now compute TF-IDF...')
    print(f'Drugs {list(testimonial_totals.keys())}')
    tf_idf = np.zeros((len(documents), len(word_columns)))
    for i, d in enumerate(documents):
        for j, word in enumerate(word_columns):
            tf = counts_per_document[i][word] / (1+total_counts_per_document[i])
            idf = np.log(len(documents) / (total_documents_per_word[word] + 1))
            tf_idf[i, j] = tf*idf

    return tf_idf, affinities, word_columns, selected, drugs, testimonial_totals


def pca_on_word_matrix(tf_idf, pca_components):
    pca = PCA()
    pca.fit(tf_idf)
    print(f'PCA explains {100*np.sum(pca.explained_variance_ratio_[:pca_components]):0.1f}% of variance with {pca_components} top PCA components.')
    tf_idf_reduced = pca.transform(tf_idf)[:, :pca_components]
    print(f'PCA reduces tf idf shape:{tf_idf_reduced.shape} from tf_idf shape: {tf_idf.shape}')
    plot_scree(pca_components, 100*pca.explained_variance_ratio_)
    return pca, tf_idf_reduced


def plot_scree(pca_components, percent_explained):
    _ = plt.figure(figsize=(6, 4))
    plt.plot(range(len(percent_explained)), percent_explained, 'g.-', linewidth=1)
    plt.axvline(x=pca_components, c='r', linewidth=3)
    label = f'{np.sum(percent_explained[:pca_components]):0.1f}% of variance explained by top {pca_components} of {len(percent_explained)} components'
    plt.text(pca_components+0.02*len(percent_explained), percent_explained[1], label)
    plt.title('Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('% of Variance Explained by Each Component')
    figure_path = f'results/pca_{pca_components}_of_{len(percent_explained)}_testimonials.png'
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def plot_testimonial_histogram(testimonial_totals):
    _ = plt.figure(figsize=(9, 6))
    sorted_testimonials = sorted(testimonial_totals.items(), key=lambda x: x[0])
    plt.bar(range(len(testimonial_totals)), [t[1] for t in sorted_testimonials])
    plt.xticks(np.arange(len(testimonial_totals)), [t[0] for t in sorted_testimonials], rotation=60, ha='right')
    title = f'{len(testimonial_totals)}_drugs_{sum(testimonial_totals.values())}_testimonials'
    plt.title(title.replace('_', ' '))
    plt.ylabel('Testimonials')
    plt.tight_layout()
    figure_path = f'results/histogram_{title}.png'
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def plot_meta_data(meta_data):
    f, axes = plt.subplots(len(meta_data), 2, figsize=(6, len(meta_data)*4))
    stats = Counter()
    all_ages = []
    for i, drug in enumerate(meta_data):
        ages = []
        sexes = []
        for meta in meta_data[drug]:
            try:
                ages.append(float(meta_data[drug][meta]['age']))
                all_ages.append(float(meta_data[drug][meta]['age']))
                stats['age'] += 1
                if 'male' == meta_data[drug][meta]['sex']:
                    sex = 1
                elif 'female' == meta_data[drug][meta]['sex']:
                    sex = 0
                else:
                    continue
                sexes.append(sex)
                stats['sex'] += 1
            except:
                continue
        axes[i][0].set_title(f'Meta Data for {drug}')

        axes[i][0].hist(ages, linewidth=3)
        axes[i][1].hist(sexes, linewidth=3)
        axes[i][1].set_xticks([0, 1])
        axes[i][1].set_xticklabels(['Female', 'Male'])

        print(f'Drug {drug} has mean age: {np.mean(ages):0.2f} median age: {np.median(ages)}')
    print(f'Total ages  has mean age: {np.mean(all_ages):0.2f} median age: {np.median(all_ages)}')
    axes[0][0].set_ylabel('# Testimonials')
    axes[0][0].set_xlabel('Age')
    axes[0][1].set_title('Sex')
    plt.tight_layout()
    figure_path = f'results/meta_data.png'
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)

def shuffle_and_split_data(x, y):
    assert x.shape[0] == y.shape[0]
    p = np.random.permutation(x.shape[0])
    x_shuffle = x[p]
    y_shuffle = y[p]
    x_shuffle_train = x_shuffle[x.shape[0] // 8:]
    x_shuffle_test = x_shuffle[:x.shape[0] // 8]
    y_shuffle_train = y_shuffle[x.shape[0] // 8:]
    y_shuffle_test = y_shuffle[:x.shape[0] // 8]
    return x_shuffle_train, x_shuffle_test, y_shuffle_train, y_shuffle_test


def fit_cca_and_transform(components, word_train, word_test, receptor_train, receptor_test):
    print(f'CCA on matrices {word_train.shape} and {receptor_train.shape} with {components} components.')
    cca = CCA(n_components=components, scale=False)
    cca.fit(word_train, receptor_train)
    print(f'X Score shape {cca.x_scores_.shape}, Y score shape: {cca.y_scores_.shape}')
    print(f'X Loading shape {cca.x_loadings_.shape}, Y Loading shape: {cca.y_loadings_.shape}')

    pearsons = np.array([pearsonr(x_co, y_co)[0] for x_co, y_co in zip(cca.x_scores_.T, cca.y_scores_.T)])
    print(f'Pearsons x y coefficient correlations are: {pearsons}')

    word_train_r, receptor_train_r = cca.transform(word_train, receptor_train)
    word_test_r, receptor_test_r = cca.transform(word_test, receptor_test)
    return cca, word_train_r, word_test_r, receptor_train_r, receptor_test_r


def make_affinity_map_pdsp(file):
    #receptors = big_receptors
    df = pd.read_csv(file)
    df['ligand'] = df['ligand'].str.lower()
    affinity_map = defaultdict(list)
    smile_map = defaultdict(str)
    min_val = -4.8151
    affinities = np.zeros((len(DRUGS_TO_USE), len(RECEPTORS_TO_USE)))
    for i, d in enumerate(DRUGS_TO_USE):
        for j, r in enumerate(RECEPTORS_TO_USE):
            cdf = df[(df.name == r) & (df.ligand == d)]

            if len(cdf) == 0:
                affinity_map[d].append(min_val)
                affinities[i, j] = min_val
            for v, c in cdf.SMILES.value_counts().items():
                smile_map[d] = v
                break
            for v, c in cdf.hot_ligand.value_counts().items():
                kmean = cdf[cdf.hot_ligand == v]['ki'].mean()
                kstd = cdf[cdf.hot_ligand == v]['ki'].std()
                affinity_map[d].append(-np.log10(kmean))
                affinities[i, j] = -np.log10(kmean)
                break
        print(f'Drug {d} has {len(affinity_map[d]) - sum([min_val == v for v in affinity_map[d]])} affinities. SMILES => {smile_map[d]}')
    for d in affinity_map:
        affinity_map[d] = np.array(affinity_map[d])

    fig, ax = plt.subplots(figsize=(24, 8), dpi=300)
    sb.heatmap(affinities, cmap=sb.diverging_palette(220, 20, as_cmap=True), cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Receptor Affinity Heatmap')
    ax.set_xlabel('Receptors')
    ax.set_ylabel('Drugs')
    ax.set_xticks(np.arange(len(RECEPTORS_TO_USE)) + 0.5)
    ax.set_yticks(np.arange(len(DRUGS_TO_USE)) + 0.5)
    ax.set_xticklabels(labels=RECEPTORS_TO_USE, ha='right', rotation=30)
    ax.set_yticklabels(labels=DRUGS_TO_USE, rotation=0)
    plt.tight_layout()
    figure_path='./affinity_heatmap.png'
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    df = pd.DataFrame.from_dict(affinity_map, orient='index', columns=RECEPTORS_TO_USE)
    df.to_csv('./drug_by_affinity_matrix.csv')
    return affinity_map, smile_map


def make_affinity_map(affinity_file, normalize):
    affinity_map = {}
    min_value = -4.8451 #-4.0 # -4.8451 # 4.0
    eps = 1e-7
    with open(affinity_file, 'r') as volumes:
        lol = list(csv.reader(volumes, delimiter=','))
        receptors = lol[0][1:]
        print(f"CSV has {len(receptors)} receptors:{receptors}")
        receptor_columns = defaultdict(list)

        for row in lol[1:]:
            drug = row[0].lower()
            values = []
            real_values = []
            for i, r in enumerate(receptors):
                if row[i+1].upper() == 'ND':
                    values.append(min_value)# values.append(0)
                elif row[i+1].upper() == 'UM':
                    values.append(min_value)
                    receptor_columns[r].append(min_value)
                else:
                    try:
                        real_values.append(float(row[i+1]))
                        values.append(float(row[i+1]))
                        receptor_columns[r].append(float(row[i+1]))
                    except:
                        print(f'Could not parse: {row[i+1]} skipping')
            if normalize == 'exponent':
                affinity_map[drug] = np.power(10, -np.array(values))
            else:
                affinity_map[drug] = (np.array(values)) - max(values) # Potency Transform
            if normalize == 'by_drug':
                mean = np.mean(real_values)
                std = np.std(real_values) + eps
                print(f'Normalizing by drug {drug} with mean:{mean:0.2f} and std:{std:0.2f}')
                drug_normalized = []
                for v in values:
                    if v == 0:
                        drug_normalized.append(0)
                    else:
                        drug_normalized.append((v-mean)/std)
                affinity_map[drug] = np.array(drug_normalized)

    if normalize == 'by_receptor':
        means = []
        stds = []
        for i, r in enumerate(receptor_columns):
            receptor_columns[r] = np.array(receptor_columns[r])
            print(f'receptor_columns {r} has {len(receptor_columns[r])}')
            means.append(np.mean(receptor_columns[r]))
            stds.append(np.std(receptor_columns[r]) + eps)
            print(f'Normalizing by receptor {r} with mean:{np.mean(receptor_columns[r]):0.2f} and std:{np.std(receptor_columns[r]):0.2f}')
        print(f'drug:{drug} len receptor_columns{len(receptor_columns)}')
        receptor_normalized = {}
        for drug in affinity_map:
            scaled_values = []
            for i, v in enumerate(affinity_map[drug]):
                if v == 0:
                    scaled_values.append(0)
                else:
                    scaled_values.append((v-means[i])/stds[i])
            receptor_normalized[drug] = np.array(scaled_values)
            print(f'drug:{drug} has {receptor_normalized[drug].shape} mean:{np.mean(receptor_normalized[drug]):0.2f} std:{np.std(receptor_normalized[drug]):0.2f}')
        return receptor_normalized, receptors

    df = pd.DataFrame.from_dict(affinity_map, orient='index', columns=receptors)
    df.to_csv('./drug_by_affinity_matrix.csv')
    return affinity_map, receptors


def split_clean(f, text, affinity_map, limit=None, stratify=None, offset_words=0, offset_testimonials=0):
    words = defaultdict(str)
    cur_words = []
    tag_pattern = re.compile(r"\((\d+)\)\s:")
    meta_data = defaultdict(dict)
    tags = []
    sex = "Unknown"
    age = None
    stats = Counter()
    for line in text.split("\n"):
        if line.startswith('"DOSE:') or line.startswith('DOSE:') and len(cur_words) > 0:
            stats[sex] += 1
            stats[age] += 1
            for t in tags:
                stats[t] += 1
            meta_data[offset_testimonials + len(words)] = {'sex': sex, 'tags': tags, 'age': age}
            words[offset_testimonials+len(words)] = ' '.join(cur_words)
            cur_words = []
            sex = "Unknown"
            age = None
            continue
        if len(tag_pattern.findall(line)) > 0:
            try:
                tags = line.split(':')[1].split(',')
                #print(f'{tags} ^ those are the tags')
            except:
                pass
            continue
        if line[:4] == 'Exp ':
            continue
        # if line.startswith('Gender:'):
        #     sex = line.replace('Gender: ', '').strip().lower()
        #     continue
        # if 'BODY WEIGHT:' in line:
        #     #stats[t]
        #     continue
        if line.startswith('Gender:'):
            sex = line.replace('Gender: ', '').strip().lower()
            continue
        if line.startswith('Age at time of experience: '):
            age = line.replace('Age at time of experience: ', '').strip()
            #rint(f'got age: {age}')
            continue
        if len(line.strip()) < MIN_LINE_CHARS or line[0] == '[' or line[0] == '"':
            continue
        if line.startswith(f.split('.')[0]):
            continue
        for t in line.split(" "):
            t = t.replace("\t", "").strip(CHARS_TO_REMOVE).lower()
            if t.strip() in SKIP:
                continue
            if '-' in t:
                continue
            if t.endswith('mg') or t.endswith('ug') or t.endswith('views') or "t+" in t:
                continue
            if len(t) > 2:
                cur_words.append(t)
    if len(cur_words) > 0:
        stats[sex] += 1
        stats[age] += 1
        # for t in tags:
        #     stats[t] += 1
        meta_data[offset_testimonials + len(words)] = {'sex': sex, 'tags': tags, 'age': age}
        words[offset_testimonials + len(words)] = ' '.join(cur_words)
    total_new_testimonials = len(words)
    if stratify is not None:
        new_words = {}
        for i in words:
            if stratify in ['young', 'old']:
                try:
                    if int(meta_data[i]['age']) < MEDIAN_AGE and stratify == 'young':
                        new_words[i] = words[i]
                    elif int(meta_data[i]['age']) >= MEDIAN_AGE and stratify == 'old':
                        new_words[i] = words[i]
                    else:
                        del meta_data[i]
                except Exception as e:
                    del meta_data[i]
                    pass
            elif meta_data[i]['sex'] == stratify:
                new_words[i] = words[i]
            else:
                del meta_data[i]
        words = new_words

    testimonial_idxs = defaultdict(list)
    if limit is not None and len(words) > limit:
        print(f'For file {f} we randomly sampled {limit} of the {len(words)} total testimonials.')
        words2keep = np.random.choice(list(words.keys()), size=limit, replace=False)
        new_words = {k: words[k] for k in words2keep}
        idxs = [(f.replace('.txt', '').lower(), (idx-offset_testimonials)) for idx in words2keep]
        words = new_words
    else:
        print(f'For file {f} we found {len(words)} total testimonials.')
        idxs = [(f.replace('.txt', '').lower(), (idx-offset_testimonials)) for idx in words]
    selected2drugs = {}
    selected2testimonials = {}
    for i, w in enumerate(sorted(list(words.keys()))):
        selected2drugs[offset_words+i] = f.replace('.txt', '') #.lower()
        selected2testimonials[offset_words+i] = w
    affinities = [affinity_map[f.replace('.txt', '').lower()] for _ in range(len(words))]
    # for k in stats:
    #     print(f' {k} has: {stats[k]}')
    return words, affinities, idxs, selected2drugs, total_new_testimonials, meta_data


def danilos_permutation_test(n_keep, x_matrix, y_matrix, n_permutations=1000, random_seed=42):
    actual_cca = CCA(n_components=n_keep, scale=False)
    actual_cca.fit(x_matrix, y_matrix)
    actual_pearsonr = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
                          zip(actual_cca.x_scores_.T, actual_cca.y_scores_.T)])

    permuted_state = np.random.RandomState(random_seed)
    permuted_pearsonr = []
    n_except = 0
    for i_iter in range(n_permutations):
        print(i_iter + 1)

        y_permuted = np.array([permuted_state.permutation(sub_row) for sub_row in y_matrix])

        # same procedure, only with permuted subjects on the right side
        try:
            permuted_cca = CCA(n_components=n_keep, scale=False)
            permuted_cca.fit(x_matrix, y_permuted)
            permuted_pearson = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
                                         zip(permuted_cca.x_scores_.T, permuted_cca.y_scores_.T)])
            permuted_pearsonr.append(permuted_pearson)
        except:
            n_except += 1
            permuted_pearsonr.append(np.zeros(n_keep))
    permuted_pearsonr = np.array(permuted_pearsonr)

    pvals = []
    for i_component in range(n_keep):
        cur_pval = (1.0 + np.sum(permuted_pearsonr[:1, 0] >= actual_pearsonr[i_component])) / n_permutations
        pvals.append(cur_pval)
    pvals = np.array(pvals)
    print('%i CCs are significant at p<0.05' % np.sum(pvals <= 0.05))
    print('%i CCs are significant at p<0.01' % np.sum(pvals <= 0.01))
    print('%i CCs are significant at p<0.001' % np.sum(pvals <= 0.001))
    print(f'P Values: {pvals} Exceptions: {n_except}')


def sum_drug_loadings2(cca, testimonial_totals):
    counts = Counter()
    loadings = {}
    last_i = 0
    for drug in testimonial_totals:
        counts[drug] += 1.0
        aggri = np.sum(cca.x_scores_[last_i:last_i+testimonial_totals[drug], :], axis=0)
        loadings[drug] = aggri
        last_i += testimonial_totals[drug]
    for drug in loadings:
        loadings[drug] /= counts[drug]

    return loadings

def sum_drug_loadings(drugs, cca):
    counts = Counter()
    loadings = {}
    for i, select in enumerate(sorted(list(drugs.keys()))):
        counts[drugs[select]] += 1.0
        if drugs[select] in loadings:
            loadings[drugs[select]] += cca.x_scores_[i, :]
        else:
            loadings[drugs[select]] = cca.x_scores_[i, :].copy()
    for drug in loadings:
        loadings[drug] /= counts[drug]

    return loadings


def analyze_components(cca, pca, pca_components, word_columns, receptors,
                       drugs, drug_folder, drug_loadings, top_words=36, top_receptors=9, name='me', limit=1):
    receptor_loadings = np.argsort(cca.y_loadings_, axis=0)
    pca_loadings = np.dot(cca.x_loadings_.T, pca.components_[:pca_components, :]).T
    word_loadings = np.argsort(pca_loadings, axis=0)
    print(f'pca_scaled {pca_loadings.shape} word loadings {word_loadings.shape} len words:{len(word_columns)} receptor_loadings {receptor_loadings.shape}')

    plot_histograms(cca, receptors, receptor_loadings, drug_loadings, f'results/histos_{name}_cca_on_{drug_folder}_pca_{pca_components}.png', 'blue', 'red')
    plot_histograms(cca, receptors, receptor_loadings, drug_loadings, f'results/histos_neg_{name}_cca_on_{drug_folder}_pca_{pca_components}.png', 'red', 'blue')
    figure_path = f'results/cloud_{name}_cca_on_{drug_folder}_{word_loadings.shape[-1]}_pca_{pca_components}_limit_{limit}.png'
    plot_clouds(cca, word_columns, receptors, pca_loadings, word_loadings, drug_loadings, receptor_loadings, figure_path, top_words, top_receptors, 'blue', 'red')
    figure_path = f'results/list_{name}_cca_on_{drug_folder}_{word_loadings.shape[-1]}_pca_{pca_components}_limit_{limit}.png'
    plot_lists(cca, word_columns, receptors, pca_loadings, word_loadings, drug_loadings, receptor_loadings, figure_path, top_words, top_receptors, 'blue', 'red')
    figure_path = f'results/cloud_neg_{name}_cca_on_{drug_folder}_{word_loadings.shape[-1]}_pca_{pca_components}_limit_{limit}.png'
    plot_clouds(cca, word_columns, receptors, pca_loadings, word_loadings, drug_loadings, receptor_loadings, figure_path, top_words, top_receptors, 'red', 'blue')
    figure_path = f'results/list_neg_{name}_cca_on_{drug_folder}_{word_loadings.shape[-1]}_pca_{pca_components}_limit_{limit}.png'
    plot_lists(cca, word_columns, receptors, pca_loadings, word_loadings, drug_loadings, receptor_loadings, figure_path, top_words, top_receptors, 'red', 'blue')
    for i in range(word_loadings.shape[-1]):
        print(f'\n\n\n~~~~~~~~~~~~~~~ Component {i} ~~~~~~~~~~~~~~~~~')
        print(f'Component {i} Highest 18 word loadings: {np.flip(np.array(word_columns)[word_loadings[:, i]][-(top_words+1):])}')
        print(f'Component {i} Highest 6 receptor loadings: {np.flip(np.array(receptors)[receptor_loadings[:, i]][-(top_receptors+1):])}')
        print(f'Highest 8 X loads {np.flip(pca_loadings[word_loadings[:, i], i][-(top_words+1):])}')
        print(f'Highest 8 Y loads {np.flip(cca.y_loadings_[receptor_loadings[:, i], i][-(top_receptors+1):])}\n')

        print(f'Component {i} Lowest 18 word loadings: {np.array(word_columns)[word_loadings[:, i]][:top_words]}')
        print(f'Component {i} Lowest 6 receptor loadings: {np.array(receptors)[receptor_loadings[:, i]][:top_receptors]}')
        print(f'Lowest 8 X loads {pca_loadings[word_loadings[:, i], i][:top_words]}')
        print(f'Lowest 8 Y loads {cca.y_loadings_[receptor_loadings[:, i], i][:top_receptors]}\n\n')


def _color(sign_dict, pos_color, neg_color, word, **kwargs):
    return neg_color if sign_dict[word] < 0 else pos_color


def plot_clouds(cca, word_columns, receptors, pca_loadings, word_loadings, drug_loadings, receptor_loadings, figure_path, top_words, top_receptors, pos_color, neg_color):
    components = word_loadings.shape[-1]
    f, axes = plt.subplots(3, components, figsize=(3*components, 12), dpi=300, gridspec_kw={'height_ratios': [5.1, 2, 3]})
    axes[0, 0].set_title(f'Word Clouds')
    axes[1, 0].set_title(f'Receptor Clouds')
    axes[2, 0].set_title(f'Drugs Associated with Components')
    for i in range(components):

        min_loading = np.min(pca_loadings[word_loadings[:, i], i])
        max_loading = np.max(pca_loadings[word_loadings[:, i], i])
        word_map = {}
        sign_map = {}
        for k in range(top_words):
            scaled_weight = _translate(pca_loadings[word_loadings[:, i], i][k], min_loading, max_loading, 24, 10)
            word = np.array(word_columns)[word_loadings[:, i]][k]
            word_map[word] = scaled_weight
            sign_map[word] = -1
            scaled_weight = _translate(pca_loadings[word_loadings[:, i], i][-(k+1)], min_loading, max_loading, 10, 24)
            word = np.array(word_columns)[word_loadings[:, i]][-(k+1)]
            word_map[word] = scaled_weight
            sign_map[word] = 1
        #print(f'got word map: {word_map}')
        wc = wordcloud.WordCloud(background_color='white')
        wc.generate_from_frequencies(word_map)
        bag = wc.recolor(color_func=partial(_color, sign_map, pos_color, neg_color))
        axes[0, i].imshow(bag)

        min_loading = np.min(cca.y_loadings_[receptor_loadings[:, i], i])
        max_loading = np.max(cca.y_loadings_[receptor_loadings[:, i], i])
        receptor_map = {}
        sign_map = {}
        for k in range(top_receptors):
            scaled_weight = _translate(cca.y_loadings_[receptor_loadings[:, i], i][k], min_loading, max_loading, 24, 10)
            receptor = np.array(receptors)[receptor_loadings[:, i]][k]
            receptor_map[receptor] = scaled_weight
            sign_map[receptor] = -1
            scaled_weight = _translate(cca.y_loadings_[receptor_loadings[:, i], i][-(k+1)], min_loading, max_loading, 10, 24)
            receptor = np.array(receptors)[receptor_loadings[:, i]][-(k+1)]
            receptor_map[receptor] = scaled_weight
            sign_map[receptor] = 1

        wc = wordcloud.WordCloud(background_color='white')
        wc.generate_from_frequencies(receptor_map)
        bag = wc.recolor(color_func=partial(_color, sign_map, pos_color, neg_color))
        axes[1, i].imshow(bag)

        axes[0, i].set_ylabel(f'Component {i}')
        axes[0, i].set_xticks(())
        axes[0, i].set_yticks(())
        axes[1, i].set_xticks(())
        axes[1, i].set_yticks(())
        axes[2, i].set_xticks(())
        axes[2, i].set_yticks(())

        maxi = 0
        mini = 9e9
        drug_list = list(drug_loadings.keys())
        component_drug_loadings = []
        for drug in drug_list:
            maxi = max(maxi, drug_loadings[drug][i])
            mini = min(mini, drug_loadings[drug][i])
            component_drug_loadings.append(float(drug_loadings[drug][i]))
        sorted_loadings = np.argsort(np.array(component_drug_loadings), axis=0)
        drug_list_sorted = np.array(drug_list)[sorted_loadings]
        drug_map = {}
        sign_map = {}
        scalar_w = 12
        for k, drug in enumerate(drug_list_sorted):
            scaled_weight = _translate(drug_loadings[drug][i], mini, maxi, -scalar_w, scalar_w)
            drug_map[drug] = max(3, abs(scaled_weight))
            sign_map[drug] = 1 if scaled_weight > 0 else -1

        wc = wordcloud.WordCloud(background_color='white')
        wc.generate_from_frequencies(drug_map)
        bag = wc.recolor(color_func=partial(_color, sign_map, pos_color, neg_color))
        axes[2, i].imshow(bag)

    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def plot_lists(cca, word_columns, receptors, pca_loadings, word_loadings, drug_loadings, receptor_loadings, figure_path, top_words, top_receptors, pos_color, neg_color):
    components = word_loadings.shape[-1]
    f, axes = plt.subplots(3, components, figsize=(5.8*components, 32), dpi=300,  gridspec_kw={'height_ratios': [5.2, 1.8, 5.0]})
    #f, axes = plt.subplots(3, components, figsize=(8 * components, 18), dpi=300, gridspec_kw={'height_ratios': [5, 1.0, 0.7]})
    axes[0, 0].set_title(f'Words')
    axes[1, 0].set_title(f'Receptors')
    axes[2, 0].set_title(f'Drugs Associated with Components')
    scalar_w = 24
    scalar_t = 12
    for i in range(components):
        for k in range(top_words):
            _text(axes[0, i], word_columns, word_loadings, pca_loadings, i, -(k+1), c=neg_color, scalar=scalar_w, std_scalar=2.25, max_words=top_words)
            _text(axes[0, i], word_columns, word_loadings, pca_loadings, i, k, c=pos_color, scalar=scalar_w, std_scalar=2.25, max_words=top_words)

        for k in range(top_receptors):
            _text(axes[1, i], receptors, receptor_loadings, cca.y_loadings_, i, -(k+1), c=neg_color, scalar=scalar_t, std_scalar=1.0, max_words=top_receptors)
            _text(axes[1, i], receptors, receptor_loadings, cca.y_loadings_, i, k, c=pos_color, scalar=scalar_t, std_scalar=1.0, max_words=top_receptors)

        axes[0, i].set_ylabel(f'Component {i}')
        axes[0, i].set_xticks(())
        axes[0, i].set_yticks(())
        axes[1, i].set_xticks(())
        axes[1, i].set_yticks(())
        axes[2, i].set_xticks(())
        axes[2, i].set_yticks(())

        maxi = 0
        mini = 9e9
        drug_list = list(drug_loadings.keys())
        component_drug_loadings = []
        for drug in drug_list:
            maxi = max(maxi, drug_loadings[drug][i])
            mini = min(mini, drug_loadings[drug][i])
            component_drug_loadings.append(float(drug_loadings[drug][i]))
        sorted_loadings = np.argsort(np.array(component_drug_loadings), axis=0)
        drug_list_sorted = np.array(drug_list)[sorted_loadings]
        scalar_d = 11
        offset_y_neg = -0.01  # -0.04
        offset_y_pos = 0.97  # 0.85
        for k, drug in enumerate(drug_list_sorted):
            scaled_weight = _translate(drug_loadings[drug][i], mini, maxi, -scalar_d, scalar_d)
            fontsize = max(4, abs(scaled_weight))
            if scaled_weight > 0:
                c = neg_color
                xpos = 0.53
                ypos = offset_y_neg + (k / len(drug_list))

            else:
                c = pos_color
                xpos = 0.03
                ypos = offset_y_pos - (k / len(drug_list))
            axes[2, i].text(xpos, ypos, drug.capitalize() if len(drug) > 4 else drug.upper(), fontsize=fontsize, c=c)

    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def plot_histograms(cca, receptors, receptor_loadings, drug_loadings, figure_path, pos_color, neg_color):
    components = receptor_loadings.shape[-1]
    f, axes = plt.subplots(2, components, figsize=(8*components, 6),  dpi=300)
    #axes[0, 0].set_title(f'Receptors')
    #axes[1, 0].set_title(f'Drugs')
    for i in range(components):
        names = []
        colors = []
        heights = []
        for k in range(len(receptors)):
            names.append(np.array(receptors)[receptor_loadings[:, i]][k])
            heights.append(abs(cca.y_loadings_[receptor_loadings[:, i], i][k]))
            colors.append(neg_color if cca.y_loadings_[receptor_loadings[:, i], i][k] < 0 else pos_color)
        axes[0, i].bar(names, heights, color=colors)
        axes[0, i].set_xticklabels(names, rotation=270, fontsize=9)
        axes[0, i].tick_params(axis='y', labelrotation=270, labelsize=7)
        axes[0, i].set_ylabel(f'Component {i}')
        axes[0, i].spines['top'].set_visible(False)
        axes[0, i].spines['right'].set_visible(False)
        axes[0, i].spines['bottom'].set_visible(False)
        axes[0, i].spines['left'].set_visible(False)
        names = []
        colors = []
        heights = []
        drug_list = list(drug_loadings.keys())
        component_drug_loadings = []
        for drug in drug_list:
            component_drug_loadings.append(float(drug_loadings[drug][i]))
        sorted_loadings = np.argsort(np.array(component_drug_loadings), axis=0)[::-1]
        drug_list_sorted = np.array(drug_list)[sorted_loadings]
        for k, drug in enumerate(drug_list_sorted):
            names.append(drug.capitalize() if len(drug) > 4 else drug.upper())
            heights.append(100.0*abs(drug_loadings[drug][i]))
            colors.append(neg_color if drug_loadings[drug][i] < 0 else pos_color)

        axes[1, i].bar(names, heights, color=colors)
        axes[1, i].set_xticklabels(names, rotation=90, fontsize=9)
        axes[1, i].tick_params(axis='y', labelrotation=90, labelsize=7)
        axes[1, i].spines['top'].set_visible(False)
        axes[1, i].spines['right'].set_visible(False)
        axes[1, i].spines['bottom'].set_visible(False)
        axes[1, i].spines['left'].set_visible(False)
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def _text(ax, words, loadings, ccas, i, k, c, scalar, std_scalar, max_words):
    word = np.array(words)[loadings[:, i]][k]
    min_loading = np.min(ccas[loadings[:, i], i])
    max_loading = np.max(ccas[loadings[:, i], i])
    mean_loading = np.mean(ccas[loadings[:, i], i])
    std_loading = np.std(ccas[loadings[:, i], i])
    cur_loading = ccas[loadings[:, i], i][k]
    max_words += 1
    if abs(cur_loading - mean_loading) > std_loading * std_scalar:
        scaled_weight = _translate(ccas[loadings[:, i], i][k], min_loading, max_loading, -scalar, scalar)
        fontsize = max(7, abs(scaled_weight))
        xpos = 0.52 if k < 0 else 0.02
        y_offset = (1-(1.3/max_words))
        ypos = y_offset - (abs(k+1)/max_words) if k < 0 else y_offset - (k/max_words)
        ax.text(xpos, ypos, word.capitalize(), fontsize=fontsize, c=c, alpha=0.7)


def _translate(val, cur_min, cur_max, new_min, new_max):
    val -= cur_min
    val /= (1e-5+(cur_max - cur_min))
    val *= (new_max - new_min)
    val += new_min
    return val


def plot_cca_cross_correlations2(ccas, cca_components, cca_names, pca_components, limit, selections):
    fig, axes = plt.subplots(1, 3, figsize=(9, 6))
    heatmap = np.zeros((len(ccas), len(ccas), 3))
    for i, cca1 in enumerate(ccas):
        for j, cca2 in enumerate(ccas):
            reverse_j = {v: k for k, v in selections[j].items()}
            reverse_i = {v: k for k, v in selections[i].items()}
            overlap_i = [iii for iii in selections[i] if selections[i][iii] in reverse_j]
            overlap_j = [jjj for jjj in selections[j] if selections[j][jjj] in reverse_i]
            print(f' len overlap i {len(overlap_i)} len overlap j {len(overlap_j)} ')
            for ii in range(cca_components):
                for jj in range(cca_components):
            # if i != j:
                    cca1_x_overlap = cca1.x_scores_[overlap_i]
                    cca1_y_overlap = cca1.y_scores_[overlap_i]
                    cca2_x_overlap = cca2.x_scores_[overlap_j]
                    cca2_y_overlap = cca2.y_scores_[overlap_j]
                    rho_xx = np.corrcoef(cca1_x_overlap[:, ii], cca2_x_overlap[:, jj])[1, 0]  # corrcoef returns full covariance matrix
                    rho_xy = np.corrcoef(cca1_x_overlap[:, ii], cca2_y_overlap[:, jj])[1, 0]  # corrcoef returns full covariance matrix
                    rho_yy = np.corrcoef(cca1_y_overlap[:, ii], cca2_y_overlap[:, jj])[1, 0]  # corrcoef returns full covariance matrix
                    # rho_xx = np.corrcoef(cca1.x_loadings_[:, ii], cca2.x_loadings_[:, jj])[1, 0]  # corrcoef returns full covariance matrix
                    # rho_xy = np.corrcoef(cca1.x_loadings_[:, ii], cca2.y_loadings_[:, jj])[1, 0]  # corrcoef returns full covariance matrix
                    # rho_yy = np.corrcoef(cca1.y_loadings_[:, ii], cca2.y_loadings_[:, jj])[1, 0]  # corrcoef returns full covariance matrix
                    print(f'cca{i} and cca{j} have xx {rho_xx} xy {rho_xy} and yy {rho_yy} '
                          f'cca1 xshape:{cca1.x_scores_.shape} cca2 xshape:{cca2.x_scores_.shape} ')
                    if ii == jj:
                        heatmap[i, j, 0] += np.abs(rho_xx) / cca_components
                        heatmap[i, j, 1] += np.abs(rho_xy) / cca_components
                        heatmap[i, j, 2] += np.abs(rho_yy) / cca_components

    for i, scores_correlated in zip(range(heatmap.shape[-1]), ['xx', 'xy', 'yy']):
        im = axes[i].imshow(heatmap[..., i], cmap='plasma')
        im.set_clim(0, 1)
        axes[i].set_title(f'{scores_correlated} scores correlated')
        axes[i].set_xticks(())
        axes[i].set_yticks(())
        fig.colorbar(im, ax=axes[i])
    axes[0].set_yticks(range(len(cca_names)))
    axes[0].set_yticklabels(cca_names, size='small')
    plt.tight_layout()
    figure_path = f'results/cross_correlate_{len(ccas)}_runs_{cca_components}_ccas_{pca_components}_pcas_limit_{limit}.png'
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def receptor_cca_weights_graphic(affinity_map, receptors, cca):
    components = cca.y_loadings_.shape[-1]
    sides = int(np.ceil(np.sqrt(len(receptors))))
    f, axes = plt.subplots(sides, sides, figsize=(3 * sides, 3 * sides), sharey=True)
    pos_labels = ['Perceptual', 'Auditory', 'Visual', 'Vibrance', 'Comfort', 'Terror', 'Therapy', 'Nausea']
    neg_labels = ['Mystical', 'Social', 'Emotion', 'Habit', 'Natural', 'Euphoria', 'Setting', 'Feeling']
    for i, r in enumerate(receptors):
        pos = {}
        neg = {}
        labels = []
        for j,load in enumerate(cca.y_loadings_[i, :]):
            if load > 0:
                pos[j] = load
                labels.append(pos_labels[j])
            else:
                neg[j] = load
                labels.append(neg_labels[j])
        axes[i%sides,i//sides].bar(pos.keys(), pos.values(), color = 'r')
        axes[i % sides, i // sides].bar(neg.keys(), -1*np.array(list(neg.values())), color='b')
        axes[i % sides, i // sides].set_title(r)
        axes[i % sides, i // sides].set_xlabel('Components')
        axes[i % sides, i // sides].set_xticks(range(components))
        axes[i % sides, i // sides].set_xticklabels(labels, rotation=30, ha='right')
    plt.tight_layout()
    figure_path = f'./results/receptors_{components}.png'
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def plot_drugs(word_loadings, drug_loadings):
    sides = int(np.ceil(np.sqrt(len(drug_loadings))))
    components = word_loadings.shape[-1]
    f, axes = plt.subplots(sides, sides, figsize=(3*sides, 3*sides), sharey=True)
    pos_labels = ['Perceptual', 'Auditory', 'Visual', 'Vibrance', 'Comfort', 'Terror', 'Therapy', 'Nausea']
    neg_labels = ['Mystical', 'Social', 'Emotion', 'Habit', 'Natural', 'Euphoria', 'Setting', 'Feeling']

    for i,drug in enumerate(drug_loadings):
        pos = {}
        neg = {}
        labels = []
        for j,load in enumerate(drug_loadings[drug]):
            if load > 0:
                pos[j] = load
                labels.append(pos_labels[j])
            else:
                neg[j] = load
                labels.append(neg_labels[j])
        axes[i % sides, i // sides].bar(pos.keys(), pos.values(), color='r')
        axes[i % sides, i // sides].bar(neg.keys(), -1 * np.array(list(neg.values())), color='b')
        axes[i % sides, i // sides].set_title(drug)
        axes[i % sides, i // sides].set_xlabel('Components')
        axes[i % sides, i // sides].set_xticks(range(components))
        axes[i % sides, i // sides].set_xticklabels(labels, rotation=30, ha='right')
    plt.tight_layout()
    figure_path = f'./results/drugs_{components}.png'
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def receptor_cca_drug_correlations(affinity_map, receptors, cca):
    pearsons = np.zeros((len(affinity_map), cca.y_loadings_.shape[-1]))
    drugs_ordered = []
    for i, drug in enumerate(affinity_map):
        for j, receptor_loads in enumerate(cca.y_loadings_.T):
            p = pearsonr(receptor_loads, affinity_map[drug])[0]
            pearsons[i, j] = p
        drugs_ordered.append(drug)
    indexes = np.argsort(pearsons, axis=0)
    component_indexes = np.argsort(pearsons, axis=1)
    print(f'Indexes are {indexes.shape} pearsons are: {pearsons.shape}, component_indexes {component_indexes.shape} Drugs are: {drugs_ordered}')
    for cca_component in range(cca.y_loadings_.shape[-1]):
        print(f'CCA Component {cca_component} has drug receptor correlation order:\n \t {np.array(drugs_ordered)[indexes[:, cca_component]]}')
    for i, drug in enumerate(affinity_map):
        print(f'Drug {drug} has component order:\n \t {component_indexes[i, :]} {pearsons[i, component_indexes[i, :]]}')


def plot_cca_cross_correlations(components, word_train_r, word_test_r, receptor_train_r, receptor_test_r):
    f, axes = plt.subplots(components, components, figsize=(components*2, components*2))
    # All against all cross correlation of the reconstructions
    for i in range(components):
        for j in range(components):
            axes[i, j].scatter(word_train_r[:, i], receptor_train_r[:, j], label="train", marker="*", c="b", alpha=0.4)
            axes[i, j].scatter(word_test_r[:, i], receptor_test_r[:, j], label="test", marker=".", c="r", alpha=0.4)
            axes[i, j].set_title(f'CV{i}xCV{j} ρ:{np.corrcoef(word_test_r[:, i], receptor_test_r[:, j])[0, 1]:0.2f}')
            axes[i, j].set_xticks(())
            axes[i, j].set_yticks(())
    axes[0, 0].legend(loc="best")
    figure_path = f'results/erowid_cca_{components}_cross.png'
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


if __name__ == '__main__':
    run()  # back to the top

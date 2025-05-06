import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)


def plot_embeddings(embeddings, labels, title, filename):
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(
        np.linspace(
            start=0, stop=1, num=len(unique_labels)
        )
    )

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)[0]
        plt.scatter(
            embeddings[idx, 0],
            embeddings[idx, 1],
            label=f'Class {label}',
            alpha=0.5,
            color=colors[i]
        )

    plt.legend()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(filename, format='svg')
    plt.close()


def main():
    original_model = SentenceTransformer(
        "/workspace/research/HuggingfaceCKPTs/multi-qa-distilbert-cos-v1"
    )
    finetuned_model = SentenceTransformer(
        "/workspace/research/TextStructureAlign/_ckpt/cora/st_models/ablate-reg-1-reg"
    )

    dataset = torch.load("/workspace/research/TextStructureAlign/data/raw/cora/cora.pt")

    sentences = dataset.raw_texts
    labels_np = np.array(dataset.y)

    original_embeddings = original_model.encode(sentences)
    finetuned_embeddings = finetuned_model.encode(sentences)

    all_embeddings = np.vstack((original_embeddings, finetuned_embeddings))

    pca = PCA(
        n_components=50
    )
    reduced_embeddings = pca.fit_transform(all_embeddings)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=300
    )
    reduced_embeddings = tsne.fit_transform(reduced_embeddings)

    original_reduced = reduced_embeddings[:len(sentences), :]
    finetuned_reduced = reduced_embeddings[len(sentences):, :]

    plot_embeddings(
        embeddings=original_reduced,
        labels=labels_np,
        title='Original Model Embeddings',
        filename='original_embeddings.svg'
    )
    plot_embeddings(
        embeddings=finetuned_reduced,
        labels=labels_np,
        title='Finetuned Model Embeddings',
        filename='finetuned_embeddings.svg'
    )


if __name__ == "__main__":
    main()

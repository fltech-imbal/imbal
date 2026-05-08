# t-SNE Model Explanation

Our t-SNE model explanation implementations are a wrapper around the scikit-learn *t-Distributed Stochastic Neighbor
Embeddings* (t-SNE) implementation, which can be found
[here](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). The original paper where t-SNE was
proposed can be found [here](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).

These resources should be used as reference over our documentation wherever an explanation of the
inner functionalities of t-SNE are necessary. Our goal is not to provide significant
expansions to t-SNE's original functionalities, nor provide further insight into t-SNE's
inner workings, but rather to simply provide a simpler, more streamlined interface
for some of t-SNE's functionalities.

Below is a list of the functions we have implement which utilize t-SNE's capabilities:
- [imbal.classification.tsne_visualization](../imbal/classification/tsne_visualization.md)
- [imbal.regression.tsne_visualization](../imbal/regression/tsne_visualization.md)
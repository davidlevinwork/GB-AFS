import pandas as pd
from sklearn.manifold import TSNE


class DimensionReductionService:
    @staticmethod
    def tsne(F: pd.DataFrame, n_components: int = 2, perplexity: int = 30, n_iter: int = 1000) -> pd.DataFrame:
        """Perform dimension reduction using the TSNE algorithm

        Parameters
        ----------
        F : pandas.DataFrame
            Matrix F with separation values for each feature
        n_components : int, optional
            Number of components parameter for TSNE, by default 2
        perplexity : int, optional
            Perplexity parameter for TSNE, by default 30
        n_iter : int, optional
            Number of iterations for TSNE, by default 1000

        Returns
        -------
        pandas.DataFrame
            Reduced matrix F
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        F_reduced = tsne.fit_transform(F)
        return F_reduced

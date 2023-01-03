import time

import pandas as pd
from sklearn.manifold import TSNE


class DimensionReductionService:
    def __init__(self, log_service, visualization_service):
        self.log_service = log_service
        self.visualization_service = visualization_service

    def tsne(self, F: pd.DataFrame, n_components: int = 2, perplexity: int = 30, n_iter: int = 1000) -> pd.DataFrame:
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
        start = time.time()
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
        F_reduced = tsne.fit_transform(F)

        self.visualization_service.plot_tsne(F_reduced, 'Jeffries-Matusita')

        end = time.time()
        self.log_service.log('Info', f'[Dimension Reduction Service] : Number of components: ({n_components}) * '
                                     f'Perplexity: ({perplexity}) * Number of iterations: ({n_iter})')
        self.log_service.log('Debug', f'[Dimension Reduction Service] : Total run time in seconds: '
                                      f'[{round(end-start, 3)}]')
        return F_reduced

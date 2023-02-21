import time

import pandas as pd
from sklearn.manifold import TSNE


class DimensionReductionService:
    def __init__(self, log_service, visualization_service):
        self.log_service = log_service
        self.visualization_service = visualization_service

    def tsne(self, F: pd.DataFrame, stage: str, fold_index: int, n_components: int = 2, perplexity: int = 30,
             n_iter: int = 1000) -> pd.DataFrame:
        """Execute dimensionality reduction (using the TSNE algorithm)

        Parameters
        ----------
        F : pandas.DataFrame
            JM Matrix (with separation values for each feature)
        stage: str
            Stage of the algorithm (Train, Full Train, Test)
        fold_index: int
            Index of the given k-fold (for saving results & plots)
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
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        F_reduced = tsne.fit_transform(F)

        self.visualization_service.plot_tsne(F_reduced, stage, fold_index)

        end = time.time()
        self.log_service.log('Info', f'[Dimension Reduction Service] : Number of components: ({n_components}) * '
                                     f'Perplexity: ({perplexity}) * Number of iterations: ({n_iter})')
        self.log_service.log('Debug', f'[Dimension Reduction Service] : Total run time in seconds: '
                                      f'[{round(end-start, 3)}]')
        return F_reduced

import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from Model.LogService.Log import log_service
from Model.VisualizationService.Visualization import visualization_service


class DimensionReductionService:
    @staticmethod
    def tsne(F: np.ndarray, stage: str, fold_index: int, n_components: int = 2, perplexity: float = 30.0,
             n_iter: int = 1000) -> np.ndarray:
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
        perplexity : float, optional
            Perplexity parameter for TSNE, by default 30.0
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

        visualization_service.plot_tsne(F_reduced, stage, fold_index)

        end = time.time()
        log_service.log('Info', f'[Dimension Reduction Service] : Number of components: ({n_components}) * '
                                f'Perplexity: ({perplexity}) * Number of iterations: ({n_iter})')
        log_service.log('Debug', f'[Dimension Reduction Service] : Total run time in seconds: '
                                 f'[{round(end - start, 3)}]')
        return F_reduced

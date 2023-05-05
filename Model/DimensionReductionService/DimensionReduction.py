import time
import numpy as np
from config import config
from sklearn.manifold import TSNE
from Model.LogService.Log import log_service
from Model.VisualizationService.Visualization import visualization_service


class DimensionReductionService:
    @staticmethod
    def tsne(F: np.ndarray, stage: str, fold_index: int) -> np.ndarray:
        """Execute dimensionality reduction (using the TSNE algorithm)

        Parameters
        ----------
        F : pandas.DataFrame
            JM Matrix (with separation values for each feature)
        stage: str
            Stage of the algorithm (Train, Full Train, Test)
        fold_index: int
            Index of the given k-fold (for saving results & plots)

        Returns
        -------
        pandas.DataFrame
            Reduced matrix F
        """
        start = time.time()
        tsne = TSNE(n_iter=config.t_sne.n_iter,
                    perplexity=config.t_sne.perplexity,
                    n_components=config.t_sne.n_components)
        F_reduced = tsne.fit_transform(F)

        visualization_service.plot_tsne(F_reduced, stage, fold_index)

        end = time.time()
        log_service.log('Info', f'[Dimension Reduction Service] : '
                                f'Perplexity: ({config.t_sne.perplexity}) * '
                                f'Number of iterations: ({config.t_sne.n_iter}) * '
                                f'Number of components: ({config.t_sne.n_components})')
        log_service.log('Debug', f'[Dimension Reduction Service] : Total run time in seconds: '
                                 f'[{round(end - start, 3)}]')
        return F_reduced

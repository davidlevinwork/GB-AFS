# GB-AFS: Graph-Based Automatic Feature Selection for Multi-Class Classification via Mean Simplified Silhouette
Official implementation of the Graph-Based Automatic Feature Selection (GB-AFS) model, a novel filter-based feature selection method for multi-class classification tasks. GB-AFS automatically determines the minimum combination of features required to sustain prediction performance while maintaining complementary discriminating abilities between different classes, without the need for user-defined parameters.

![](Resources/GB-AFS.png)

## Abstract
This repo introduces a novel graph-based filter method for automatic feature selection (abbreviated as GB-AFS) for multi-class classification tasks. The method determines the minimum combination of features required to sustain prediction performance while maintaining complementary discriminating abilities between different classes. It does not require any user-defined parameters such as the number of features to select. The methodology employs the Jeffries--Matusita (JM) distance in conjunction with t-distributed Stochastic Neighbor Embedding (t-SNE) to generate a low-dimensional space reflecting how effectively each feature can differentiate between each pair of classes. The minimum number of features is selected using our newly developed Mean Simplified Silhouette (abbreviated as MSS) index, designed to evaluate the clustering results for the feature selection task. Experimental results on public data sets demonstrate the superior performance of the proposed GB-AFS over other filter-based techniques and automatic feature selection approaches. Moreover, the proposed algorithm maintained the accuracy achieved when utilizing all features, while using only $7%$ to $30%$ of the features. Consequently, this resulted in a reduction of the time needed for classifications, from $15%$ to $70%$.

## Setup Environment

```bash
git clone https://github.com/davidlevinwork/GB-AFS.git
cd GB-AFS
pip install -r requirements.txt
```
Make sure to fill in the ***config.yaml*** file before running the main script:
```bash
python3 main.py
```

## Configuration File Explanation
This section provides explanations for each parameter in the configuration file. 
<br /> <br />
Before running the model, you need to fill in the ***config.yaml*** file, which contains important configurations required for the code to run properly. This file should be located in the root folder of the project. Please make sure to set the appropriate values for your specific use case. 

### *mode*
Determines the run mode for the algorithm. There are two options:
- `basic`: Runs the GB-AFS model only (finds the features' subset)
- `full`: Runs the GB-AFS model in proof of concept configuration (including classification, full plots, benchmarks, and results comparison)
### *plots*
Determines the output visualizations provided to the user.
- `t_sne`: A graph illustrating feature space separation capabilities after dimension reduction
- `silhouette`: A graph depicting silhouette values as a function of K, showcasing the advantages of the MSS index ('basic' mode displays only MSS)
- `clustering`: A series of graphs for each K value in the range [2, Number of features] representing clusters in the new feature space using distinct colors
- `clustering_based_jm`: A series of graphs for each K value in the range [2, Number of features] representing clusters in the new feature space, with colors based on average JM values
- `accuracy_to_silhouette`: A graph showing the correlation between accuracy and MSS values for all K values ('basic' mode does not display accuracy)
### *dataset*
Dataset-related parameters.
- `dataset_path`: Relative path to the dataset file
- `train_test_split`: Format "%d-%d" to specify the train-test split ratio
- `label_column_str`: The name of the column containing the data labels
### *k_fold*
K-fold cross-validation settings.
- `n_splits`: Number of splits for k-fold cross-validation
- `shuffle`: Whether to shuffle the data before splitting it into folds
### *t_sne*
t-SNE algorithm settings.
- `n_iter`: Number of iterations for optimization
- `perplexity`: The perplexity value must be LOWER than the number of features in the given dataset
### *k_medoids*
K-Medoids algorithm settings.
- `method`: The algorithm to use. Options are `alternate` (faster) or `pam` (more accurate)


## Support
If you find this repository useful, we would appreciate it if you could give it a STAR. Your support helps us to continue improving the model and its implementation.

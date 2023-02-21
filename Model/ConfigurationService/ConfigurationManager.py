import json


class ConfigManager:
    def __init__(self, config_file_path):
        with open(config_file_path) as f:
            self.config = json.load(f)

    @property
    def get_dataset_name(self):
        return self.config['dataset']['name']

    @property
    def get_dataset_path(self):
        return self.config['dataset']['path']

    @property
    def get_dataset_labels_col_name(self):
        return self.config['dataset']['labels_column_name']

    @property
    def get_algo_k_fold(self):
        return self.config.get('algorithm_properties')['KFold']

    @property
    def get_data_val_ratio(self):
        return self.config.get('algorithm_properties')['val_ratio']

    @property
    def get_metric(self):
        return self.config.get('algorithm_properties')['metric']

    @property
    def get_tsne_n_iter(self):
        return self.config.get('t_SNE')['n_iter']

    @property
    def get_tsne_perp(self):
        return self.config.get('t_SNE')['perplexity']

    @property
    def get_tsne_n_comp(self):
        return self.config.get('t_SNE')['n_components']

    @property
    def get_kMedoid_method(self):
        return self.config.get('K_Medoid')['method']

    @property
    def get_kMedoid_init(self):
        return self.config.get('K_Medoid')['init']

from dataset import gen_random_loaders


class BaseNodes:
    def __init__(self, data_name, n_nodes, batch_size, classes_per_node):
        self.data_name = data_name
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.classes_per_node = classes_per_node
        self._init_dataloaders()

    def _init_dataloaders(self):
        self.dataloaders, self.total_num_points, self.num_points_p_c = gen_random_loaders(
            self.data_name,
            self.n_nodes,
            self.batch_size,
            self.classes_per_node
        )

        self.train_loaders, self.val_loaders, self.test_loaders = self.dataloaders

    def __len__(self):
        return self.n_nodes

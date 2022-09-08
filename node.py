from dataset import gen_random_loaders


class BaseNodes:
    def __init__(self, data_name, n_nodes, batch_size, classes_per_node):
        self.data_name = data_name
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.classes_per_node = classes_per_node
        self._init_dataloaders()

    def _init_dataloaders(self):
        self.train_loaders, self.val_loaders, self.test_loaders = gen_random_loaders(
            self.data_name,
            self.n_nodes,
            self.batch_size,
            self.classes_per_node
        )

    def __len__(self):
        return self.n_nodes

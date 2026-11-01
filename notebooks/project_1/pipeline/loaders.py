"""DataLoaders: feature matrix + targets -> batches. Takes care of Training Batching and Loading."""
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset, TensorDataset

# @todo this has to be optimized or at least it has to be documented how its optimized with badge sizing and concurrency, its the one lecture i missed
class SparseDataset(Dataset):
    """Serves (features, label) pairs from a sparse tf-idf matrix.

    The full dense matrix would be examples x (4 * vocabulary) floats and not fit in memory,
    so a row is only densified when the DataLoader asks for it.
    """

    def __init__(self, X, y):
        self.X = X  # scipy csr matrix, one row per example
        self.y = torch.tensor(y, dtype=torch.long)  # 0 -> hypothesis_1, 1 -> hypothesis_2

    def __len__(self):
        # DataLoader uses this to know how many examples exist per epoch
        return self.X.shape[0]

    def __getitem__(self, index):
        # sparse row -> dense 1d float tensor of size 4 * vocabulary
        x = torch.tensor(self.X[index].toarray().ravel(), dtype=torch.float32)
        return x, self.y[index]


def make_loader(X, y, batch_size, shuffle):
    """Dense numpy matrix -> TensorDataset, sparse scipy matrix -> SparseDataset. Only the training data is shuffled.

    DataLoader draws batch_size indices, looks up each example and stacks the results into
    x: (batch_size, input_dim) and y: (batch_size,).
    """
    if sparse.issparse(X):
        dataset = SparseDataset(X, y)
    else:
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

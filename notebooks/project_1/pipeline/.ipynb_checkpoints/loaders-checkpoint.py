"""DataLoaders: feature matrix + targets -> batches. Takes care of Training Batching and Loading."""
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset, TensorDataset

class SparseDataset(Dataset):
    """Serves (features, label) pairs from a sparse tf-idf matrix, a row is only made dense when the DataLoader asks for it.

    Why sparse: a tf-idf row has one column per vocabulary word and sentence (4 * 16,189 = 64,756), but one
    example only contains ~34 words, so 99.95% of its values are 0.
    A csr matrix stores only the non-zero values and their columns, e.g. row 0 ("a baby panda was lost ..."):
        data    = [0.14, 0.40, 0.21, 0.37, ...]   tf-idf weights of 'a', 'baby', 'in', 'lost'
        indices = [227,  1206, 7159, 8549, ...]   their columns, every other column is 0
    X_train takes 6.8 MB as csr instead of 4.35 GB dense as float32 (8.70 GB as float64, what tf-idf returns).

    GloVe features don't need this: all of their values are non-zero, so they stay a dense TensorDataset.
    The model still gets dense rows: one batch of 64 is 16.6 MB, only 2,108 of its 4.1 million values are non-zero.

    The memory savings are thanks to the coding agent, i myself wouldn't think that far.
    """

    def __init__(self, X, y):
        self.X = X  # scipy csr matrix, one row per example
        self.y = torch.tensor(y, dtype=torch.long)  # 0 -> hypothesis_1, 1 -> hypothesis_2

    def __len__(self):
        # DataLoader uses this to know how many examples exist per epoch
        return self.X.shape[0]

    def __getitem__(self, index):
        # one csr row -> dense float32 tensor with all 64,756 values (zeros filled in), only for this one example
        x = torch.tensor(self.X[index].toarray().ravel(), dtype=torch.float32)
        return x, self.y[index]


def make_loader(X, y, batch_size, shuffle):
    if sparse.issparse(X):
        dataset = SparseDataset(X, y)
    else:
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

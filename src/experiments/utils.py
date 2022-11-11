from jax import jit
from neural_tangents import stax
import neural_tangents as nt
from utils.coresets import BilevelCoreset
import builder
import torch

_, _, kernel_fn = stax.serial(
    stax.Conv(32, (5, 5), (1, 1), padding='SAME', W_std=1., b_std=0.05),
    stax.Relu(),
    stax.Conv(64, (5, 5), (1, 1), padding='SAME', W_std=1., b_std=0.05),
    stax.Relu(),
    stax.Flatten(),
    stax.Dense(128, 1., 0.05),
    stax.Relu(),
    stax.Dense(10, 1., 0.05))
kernel_fn = jit(kernel_fn, static_argnums=(2,))


def generate_cnn_ntk(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    K = np.zeros((n, m))
    for i in range(m):
        K[:, i:i + 1] = np.array(kernel_fn(X, Y[i:i + 1], 'ntk'))
    return K


def cross_entropy(K, alpha, y, weights, lmbda):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
    if lmbda > 0:
        loss_value += lmbda * \
            torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss_value


if __name__ == '__main__':
    def proxy_kernel_fn(x, y): return generate_cnn_ntk(
        x.view(-1, 28, 28, 1).numpy(), y.view(-1, 28, 28, 1).numpy())
    print(proxy_kernel_fn)

    limit = 2500

    dataset = builder.make_dataset({
        'name': 'MNIST',
        'batch_size': 256,
        'collapse_targets': False
    })['train']
    print(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=limit, shuffle=False)
    X, y = next(iter(loader))

    bc = bilevel_coreset.BilevelCoreset(
        outer_loss_fn=cross_entropy,
        inner_loss_fn=cross_entropy,
        out_dim=10,
        candidate_batch_size=1000,
        max_outer_it=1)
    coreset_inds, _ = bc.build_with_representer_proxy_batch(
        X, y, subset_size, proxy_kernel_fn, cache_kernel=True, start_size=1, inner_reg=1e-7)

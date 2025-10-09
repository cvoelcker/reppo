import tqdm
import jax
import jax.numpy as jnp
from jax import jacobian
import distrax

# User must provide:
# E(x): scalar energy
# log_prior(x): scalar log density of prior
# G_fn(x): returns SPD matrix G(x) (shape (d,d))
# lambda_scalar: positive scalar
# rng: jax.random.PRNGKey

rotation_45 = jnp.array([
    [jnp.cos(jnp.pi/4 + 0.2), -jnp.sin(jnp.pi/4 + 0.2)],
    [jnp.sin(jnp.pi/4 + 0.2),  jnp.cos(jnp.pi/4 + 0.2)]
])

def gaussian_log_prior(x, mu = jnp.zeros((2,)), cov = (jnp.array([[1.0, 0.0], [0.0, 0.1]]).dot(rotation_45.T))):
    d = x.shape[0]
    x_centered = x - mu
    cov_inv = jnp.linalg.inv(cov)
    quad = 0.5 * jnp.dot(x_centered, cov_inv @ x_centered)
    logdet = jnp.linalg.slogdet(cov)[1]
    return -0.5 * d * jnp.log(2.0 * jnp.pi) - 0.5 * logdet - quad

def hump_energy(x, centroids = jnp.array([[1.0, 1.0], [-1.4, -1.4]]), scales = jnp.array([0.6, 0.6]), weights = jnp.array([0.6, 0.6])):
    # Mixture of Gaussians energy landscape (unnormalized)
    # E(x) = - log( sum_k w_k * N(x; c_k, s_k^2 I) )
    exps = jnp.array([w * jnp.exp(-0.5 * jnp.sum((x - c)**2) / (s**2)) for (c,s,w) in zip(centroids, scales, weights)])
    return -jnp.log(jnp.sum(exps) + 1e-8)


def precond_mala_posdep(key, x0, n_steps, eps, E, log_prior, G_fn, tau=1.0, lambda_scalar=1.0):
    d = x0.shape[0]

    @jax.jit
    def log_target(x):
        return (-E(x) / tau) + jnp.log(lambda_scalar) + log_prior(x)

    grad_log = jax.jit(jax.grad(log_target))
    # helper: compute divergence of G: (div G)_i = sum_j ∂_{x_j} G_{i,j}(x)
    @jax.jit
    def div_G(x):
        # jac has shape (d, d, d) with axes (k, i, j) == ∂/∂x_k G_{i,j}
        jac = jacobian(G_fn)(x)                 # (d, d, d)
        # mask[k,i,j] = 1 if k == j else 0  -> shape (d,1,d) broadcastable
        mask = jnp.eye(d)[:, None, :]
        # sum over k and j where k==j -> sum_j jac[j, i, j]
        return jnp.sum(jac * mask, axis=(0,2))  # shape (d,)

    @jax.jit
    def log_mvnorm(x, mean, cov_chol):
        # log N(x; mean, Cov) using Cholesky of Cov
        # cov_chol is lower triangular L with Cov = L @ L.T
        y = x - mean
        # solve L @ z = y  => z = L^{-1} y
        z = jax.scipy.linalg.solve_triangular(cov_chol, y, lower=True)
        quad = 0.5 * jnp.dot(z, z)
        # logdet(Cov) = 2 * sum(log diag(L))
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(cov_chol))))
        return -0.5 * d * jnp.log(2.0 * jnp.pi) - 0.5 * logdet - quad

    x = x0
    key = key
    chain = []
    accepts = 0

    for i in tqdm.tqdm(range(n_steps)):
        def step(x, key):
            key, sk = jax.random.split(key)
            Gx = G_fn(x)
            # ensure symmetry numerically
            Gx = 0.5 * (Gx + Gx.T)
            Lx = jnp.linalg.cholesky(Gx)                # Lx @ Lx.T = Gx

            g = grad_log(x)
            divg = div_G(x)
            mu_x = x + 0.5 * eps * (Gx @ g + divg)      # proposal mean

            z = jax.random.normal(sk, shape=x.shape)
            x_prop = mu_x + jnp.sqrt(eps) * (Lx @ z)    # proposal

            # compute reverse proposal params at x_prop
            Gxp = G_fn(x_prop)
            Gxp = 0.5 * (Gxp + Gxp.T)
            Lxp = jnp.linalg.cholesky(Gxp)

            g_prop = grad_log(x_prop)
            divg_prop = div_G(x_prop)
            mu_xprop = x_prop + 0.5 * eps * (Gxp @ g_prop + divg_prop)

            # log proposal densities (full Gaussian): q(x_prop|x) and q(x|x_prop)
            cov_chol_forward = jnp.sqrt(eps) * Lx    # chol of eps * Gx: sqrt(eps)*Lx
            cov_chol_backward = jnp.sqrt(eps) * Lxp

            log_q_forward = log_mvnorm(x_prop, mu_x, cov_chol_forward)
            log_q_backward = log_mvnorm(x, mu_xprop, cov_chol_backward)

            log_t_prop = log_target(x_prop)
            log_t_curr = log_target(x)

            log_alpha = (log_t_prop - log_t_curr) + (log_q_backward - log_q_forward)
            alpha = jnp.minimum(1.0, jnp.exp(log_alpha))

            u = jax.random.uniform(sk)
            accept = (u < alpha)
            x = jnp.where(accept, x_prop, x)
            return x, key, accept
        x, key, accept = step(x, key)
        accepts += accept
        if i % 1000 == 0:
            print(f"Step {i}, Acceptance rate: {accepts / (i+1):.3f}")

        chain.append(x)

    chain = jnp.stack(chain)
    acc_rate = accepts / n_steps
    return chain, acc_rate

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    squished_covariance = jnp.array([[1.0, 0.0], [0.0, 0.01]])
    rotated_covariance = rotation_45 @ squished_covariance @ rotation_45.T

    prior_dist = distrax.MultivariateNormalFullCovariance(
        loc=jnp.zeros((2,)),
        covariance_matrix=rotated_covariance
    )
    gaussian_log_prior = jax.jit(prior_dist.log_prob)

    key = jax.random.PRNGKey(5)
    x0 = prior_dist.sample(seed=key)  # initial position
    n_steps = 500
    eps = 1.0
    tau = 1.0
    lambda_scalar = 1.0

    def G_fn(x):
        # compute Hessian of the gaussian_log_prior at x
        hessian = -jax.hessian(gaussian_log_prior)(x)
        hessian_inv = jnp.linalg.inv(hessian)
        # ensure symmetry numerically
        hessian_inv = 0.5 * (hessian_inv + hessian_inv.T)
        # make it positive definite by eigenvalue clipping
        eigvals, eigvecs = jnp.linalg.eigh(hessian_inv)
        eigvals_clipped = jnp.clip(eigvals, a_min=1e-3)
        hessian_pd = (eigvecs * eigvals_clipped) @ eigvecs.T

        # return hessian_inv * 0.1
        return 0.01 * jnp.eye(2)

    chain, acc_rate = precond_mala_posdep(
        key, x0, n_steps, eps, hump_energy, gaussian_log_prior, G_fn, tau, lambda_scalar
    )
    print(f"Acceptance rate: {acc_rate:.3f}")

    # Plot the samples and energy landscape
    plt.figure(figsize=(12, 5))
    
    # Create subplot 1: Scatter plot of samples
    plt.subplot(1, 2, 1)
    plt.scatter(chain[:,0], chain[:,1], alpha=0.6, s=10, c='blue')
    plt.title("MCMC Samples")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Create subplot 2: Energy landscape with samples overlaid
    plt.subplot(1, 2, 2)
    
    # Create a grid for the energy function
    x_range = jnp.linspace(-4, 4, 50)  # Reduced resolution for faster computation
    y_range = jnp.linspace(-4, 4, 50)
    X, Y = jnp.meshgrid(x_range, y_range)
    
    # Vectorize the hump_energy function for efficient computation
    hump_energy_vec = jax.vmap(hump_energy)
    gaussian_log_prior_vec = jax.vmap(lambda x: gaussian_log_prior(x))
    
    # Compute energy at each grid point
    positions = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    energies = - hump_energy_vec(positions) + gaussian_log_prior_vec(positions)
    Z = energies.reshape(X.shape)
    # Convert from log energy to probability density (exp of negative log energy)
    Z_prob = jnp.exp(Z)
    
    # Plot energy landscape as contours
    contour = plt.contour(X, Y, Z_prob, levels=20, alpha=0.8, cmap='viridis')
    plt.contourf(X, Y, Z_prob, levels=20, alpha=0.3, cmap='viridis')
    plt.colorbar(contour, label='Probability Density')
    
    # Overlay the samples
    plt.scatter(chain[:,0], chain[:,1], alpha=0.7, s=8, c='red', label='MCMC Samples')
    
    plt.title("Energy Landscape with MCMC Samples")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/voelcke1/reppo/langevin_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Visualization saved to langevin_visualization.png")
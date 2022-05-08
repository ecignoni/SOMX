import itertools

import jax
from jax import vmap, jit
from jax import lax
from jax import random
import jax.numpy as jnp

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# Self-Organizing Map building blocks
# =============================================================================


def som_update_prototypes(prototypes, x, topology, learning_rate, neighbor_radius):
    """
    Update the Self-Organizing Map prototypes from a single observation.
    Arguments
    ---------
    prototypes      : jnp.ndarray, (n_prototypes, feature_dim)
                      SOM prototypes.
    x               : jnp.ndarray, (feature_dim,)
                      Single observation.
    topology        : jnp.ndarray, (n_prototypes, n_dims)
                      Topology of the prototypes.
    learning_rate   : float
                      Learning rate.
    neighbor_radius : float
                      Neighbor radius in prototype (SOM) space.
    Returns
    -------
    new_prototypes  : jnp.ndarray, (n_prototypes, feature_dim)
                      Updated prototypes
    prototypes      : jnp.ndarray, (n_prototypes, feature_dim)
                      Original prototypes
    """
    # Distance of prototypes from x in feature space
    dist = jnp.sum((prototypes - x) ** 2, axis=-1) ** 0.5
    # Nearest prototype in feature space
    nearest_in_X = jnp.argsort(dist)[0]
    # Neighbors of nearest prototype in SOM space
    dist = jnp.sum((topology[nearest_in_X] - topology) ** 2, axis=-1) ** 0.5
    neighborhood = jnp.where(dist < neighbor_radius, 1, 0)
    # Update the neighborhood
    new_prototypes = (
        prototypes + learning_rate * (x - prototypes) * neighborhood[:, jnp.newaxis]
    )
    return new_prototypes, prototypes


def som_single_reconstruction_error(prototypes, x):
    """
    Reconstruction error of the Self-Organizing Map for a single observation.
    Arguments
    ---------
    prototypes : jnp.ndarray, (n_prototypes, feature_dim)
                 SOM prototypes.
    x          : jnp.ndarray, (feature_dim,)
                 Single observation.
    Returns
    -------
    error      : float
                 Reconstruction error.
    """
    dist = jnp.sum((prototypes - x) ** 2, axis=-1)
    return jnp.sort(dist)[0]


def som_reconstruction_error(prototypes, X):
    """
    Reconstruction error of the Self-Organizing Map for a bunch of observations.
    Arguments
    ---------
    prototypes : jnp.ndarray, (n_prototypes, feature_dim)
                 SOM prototypes
    X          : jnp.ndarray, (n_observations, feature_dim)
    Returns
    -------
    error      : float
                 Reconstruction error over the whole data.
    """
    error = vmap(lambda x: som_single_reconstruction_error(prototypes, x))(X)
    return jnp.sum(error)


def som_get_topology(prototypes_shape):
    """
    Generate the topology of the Self-Organizing Map given the shape of the prototypes.
    Arguments
    ---------
    prototypes_shape : jnp.ndarray, (shape0, shape1, ...)
                       Prototypes shape
    Returns
    -------
    topology         : jnp.ndarray, (n_prototypes, len(prototypes_shape))
                       Topology of the Self-Organizing Map.
    """
    topology = jnp.array(list(itertools.product(*(range(n) for n in prototypes_shape))))
    return topology


def som_n_prototypes(prototypes_shape):
    """
    Get the total number of prototypes of the Self-Organizing Map.
    Arguments
    ---------
    prototypes_shape : jnp.ndarray, (shape0, shape1, ...)
                       Prototypes shape
    Returns
    -------
    n_prototypes     : int
                       Total number of prototypes.
    """
    n_prototypes = jnp.prod(prototypes_shape)
    return n_prototypes


def som_gaussian_initialize(prototypes_shape, X, key):
    """
    Gaussian initialization of the Self-Organizing Map prototypes.
    Arguments
    ---------
    prototypes_shape : jnp.ndarray, (shape0, shape1, ...)
                       Prototypes shape.
    X                : jnp.ndarray, (n_observations, feature_dim)
                       Observations.
    key              : jax.random.KeyArray
                       JAX random PRNGKey
    Returns
    -------
    prototypes       : jnp.ndarray, (n_prototypes, feature_dim)
                       SOM prototypes.
    key              : jax.random.KeyArray
                       Newly generated JAX random PRNGKey
    """
    n_prototypes = som_n_prototypes(prototypes_shape)
    mu_x = jnp.mean(X, axis=0)
    std_x = jnp.mean(X, axis=0)
    key, key_init = random.split(key)
    prototypes = random.multivariate_normal(
        key_init, mean=mu_x, cov=jnp.diag(std_x), shape=(n_prototypes,)
    )
    return prototypes, key


@jit
def som_step(X, prototypes, topology, learning_rate, neighbor_radius):
    """
    One Self-Organizing Map online update over the whole data `X`.
    Arguments
    ---------
    X               : jnp.ndarray, (n_observations, feature_dim)
                      Observations.
    prototypes      : jnp.ndarray, (n_prototypes, feature_dim)
                      SOM prototypes.
    topology        : jnp.ndarray, (n_prototypes, n_dims)
                      SOM topology.
    learning_rate   : float
                      Learning rate.
    neighbor_radius : float
                      Neighbor radius in SOM space.
    Returns
    -------
    prototypes      : jnp.ndarray, (n_observations, feature_dim)
                      Updated observations.
    """

    def inner_func(i, prototypes):
        prototypes, old_prototypes = som_update_prototypes(
            prototypes, X[i], topology, learning_rate, neighbor_radius
        )
        return prototypes

    lower = 0
    upper = X.shape[0]
    # Loop in LAX to compile it faster
    prototypes = lax.fori_loop(lower, upper, inner_func, prototypes)
    return prototypes


# =============================================================================
# Self-Organizing Map Models
# =============================================================================


def som_online(X, prototypes_shape, learning_rate, neighbor_radius, key, n_rounds):
    """
    Standard Self-Organizing Map with online updates.
    Arguments
    ---------
    X                : jnp.ndarray, (n_observations, feature_dim)
                       Observations.
    prototypes_shape : jnp.ndarray, (shape0, shape1, ...)
                       Prototypes shape.
    learning_rate    : float
                       Learning rate.
    neighbor_radius  : float
                       Neighbor radius in SOM space.
    key              : jax.random.KeyArray
                       JAX random PRNGKey
    n_rounds         : int
                       Number of training rounds (epochs)
    Returns
    -------
    results          : dict
                       Self-Organizing Map results
                       "topology"   : SOM topology
                       "prototypes" : Final SOM prototypes
                       "history"    : History of SOM prototypes (end of each epoch)
                       "reconstruction_error" : History of reconstruction errors (end of each epoch)

    """
    prototypes_shape = jnp.array(prototypes_shape)
    n_prototypes = som_n_prototypes(prototypes_shape)

    # Instantiate the topology
    topology = som_get_topology(prototypes_shape)

    # Initialize the prototypes
    prototypes, key = som_gaussian_initialize(prototypes_shape, X, key)

    # Update the prototypes
    history = [prototypes]
    reconstruction_error = [som_reconstruction_error(prototypes, X)]

    iterator = tqdm(range(n_rounds), ncols=60, desc="Epoch")
    for _ in iterator:
        prototypes = som_step(X, prototypes, topology, learning_rate, neighbor_radius)
        reconstruction_error.append(som_reconstruction_error(prototypes, X))
        history.append(prototypes)

    return dict(
        topology=topology,
        prototypes=prototypes,
        history=history,
        reconstruction_error=reconstruction_error,
    )


# =============================================================================
# Scheduler functions
# =============================================================================


def exponential_decay(param, decay_rate):
    """
    Exponential decay for `param` with rate `decay_rate`
    Arguments
    ---------
    param      : float
                 Input parameter
    decay_rate : float
                 Decay rate
    Returns
    -------
    new_param  : float
                 Updated parameter
    """
    return param * (1.0 - decay_rate / 100.0)


def linear_decay(param, decay_rate):
    """
    Linear decay for `param` with rate `decay_rate`
    Arguments
    ---------
    param      : float
                 Input parameter
    decay_rate : float
                 Decay rate
    Returns
    -------
    new_param  : float
                 Updated parameter
    """
    return param - decay_rate


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_som_space(som_topology, prototypes, X, key, color_by=None, cmap="viridis"):
    """
    Analogous to the kind of plot in Tibshirani, Figure 14.16, The Elements of Statistical Learning.
    Arguments
    ---------
    som_topology : jnp.ndarray, (n_prototypes, n_dims)
                   SOM topology.
    prototypes   : jnp.ndarray, (n_prototypes, feature_dim)
                   SOM prototypes.
    X            : jnp.ndarray, (n_observations, feature_dim)
                   Observations.
    key          : jax.random.KeyArray
                   JAX random PRNGKey
    color_by     : None or jnp.ndarray, (n_observations,)
                   If None, points are colored in red.
                   Otherwise, points are colored according to mapped values from `color_by`.
    cmap         : str
                   Matplotlib colormap name.
    """

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
    [b.set_linewidth(2) for b in ax.spines.values()]

    # Draw the circles
    circles = [
        plt.Circle(loc, 0.4, edgecolor="k", facecolor="none", lw=2.0)
        for loc in som_topology
    ]
    _ = [ax.add_patch(c) for c in circles]

    def nearest_prot_to_x(prototypes, x):
        distances = jnp.sum((prototypes - x) ** 2, axis=-1) ** 0.5
        return jnp.argsort(distances)[0]

    mapped_prots = jnp.array([nearest_prot_to_x(prototypes, x) for x in X])

    # Create an array of colors corresponding to each point
    # If no array is passed with values that are used to color
    # the points, color everything by red
    if color_by is not None:
        cmap = mpl.cm.get_cmap(cmap)
        _color_by = (color_by - jnp.min(color_by)) / (
            jnp.max(color_by) - jnp.min(color_by)
        )
        colors = cmap(_color_by)
    else:
        colors = ["r" for _ in range(len(X))]

    for i, mapped_p in enumerate(mapped_prots):
        key, key_p, key_d = random.split(key, 3)
        # Center of the circle
        center = som_topology[mapped_p]
        # Random angle in [0, 2*pi]
        theta = random.uniform(key_p, minval=0, maxval=2 * jnp.pi)
        # Random distance in [0, 1]
        dist = random.uniform(key_d)
        # Coordinates of the point to be plotted, falling inside the right circle
        coords = jnp.array(
            (
                (
                    center[0] + jnp.cos(theta) * 0.4 * dist,
                    center[1] + jnp.sin(theta) * 0.4 * dist,
                )
            )
        )
        ax.scatter(coords[0], coords[1], color=colors[i], ec="k")

    # Set the axes limits
    xmin = jnp.min(som_topology[:, 0]) - 0.5
    xmax = jnp.max(som_topology[:, 0]) + 0.5
    ymin = jnp.min(som_topology[:, 1]) - 0.5
    ymax = jnp.max(som_topology[:, 1]) + 0.5
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    return fig, ax

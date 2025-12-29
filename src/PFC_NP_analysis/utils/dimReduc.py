import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import lsqr, LinearOperator
import umap

def pca_fit(
    data: np.ndarray,
    pca_num: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    (Weighted) PCA fit that returns principal component loadings and the projection matrix.

    Parameters
    ----------
    data : ndarray, shape (T, N)
        Rows are samples/timepoints, columns are variables (neurons/features).
    pca_num : int
        Number of principal components to keep.
    weights : ndarray | None, shape (T,)
        Non-negative sample weights for each sample/timepoint (row of `data`).
        If None, defaults to uniform weights (ordinary PCA).
        Only relative scale matters; will be normalized internally.

    Returns
    -------
    pcs : ndarray, shape (N, N)
        Principal component loading vectors (columns). First k are top-k loadings.
    denoise_matrix : ndarray, shape (N, N)
        Projection matrix onto the top-k subspace: partial @ partial.T
    """
    X = np.asarray(data)
    if X.ndim != 2:
        raise ValueError(f"`data` must be 2D, got shape {X.shape}")

    T, N = X.shape
    k = int(pca_num)
    if k < 1:
        raise ValueError("`pca_num` must be >= 1")
    k = min(k, N)  # cannot exceed number of variables

    # Handle weights (per-sample)
    if weights is None:
        w = np.ones(T, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != T:
            raise ValueError(f"`weights` must have shape (T,), got {w.shape}, T={T}")
        if np.any(w < 0):
            raise ValueError("`weights` must be non-negative")
        if not np.isfinite(w).all():
            raise ValueError("`weights` must be finite")

    w_sum = float(w.sum())
    if w_sum <= 0:
        raise ValueError("Sum of `weights` must be > 0")

    # Normalize weights (scale doesn't matter)
    w = w / w_sum  # sum to 1
    sqrt_w = np.sqrt(w)  # (T,)

    # Weighted centering across samples
    mu = (w[:, None] * X).sum(axis=0, keepdims=True)  # (1, N)
    Xc = X - mu

    # Apply sqrt weights to rows (samples)
    Xw = Xc * sqrt_w[:, None]  # (T, N)

    # PCA via SVD: Xw = U S Vt; columns of V (Vt.T) are loadings in variable space
    _, _, Vt = svd(Xw, full_matrices=False)
    pcs = Vt.T  # (N, N)

    partial = pcs[:, :k]               # (N, k)
    denoise_matrix = partial @ partial.T  # (N, N)

    return pcs, denoise_matrix


def umap_fit(
    data: np.ndarray,
    n_neighbors: int = 50,
    min_dist: float = 0.1,
    n_components: int = 3,
    metric: str = "correlation",
    random_state: int = 42,
    densmap: bool = False,
    **umap_kwargs,
) -> umap.UMAP:
    """
    UMAP fit that returns low-dimensional embeddings and the fitted model.

    Parameters
    ----------
    data : ndarray, shape (T, N)
        Row is timepoints/samples, columns are variables (neurons).
    n_components : int
        Target embedding dimension.
    n_neighbors : int
        UMAP local neighborhood size.
    min_dist : float
        UMAP minimum distance between points in the embedding.
    metric : str
        Distance metric for UMAP (e.g., 'euclidean', 'cosine', ...).
    random_state : int | None
        Seed for reproducibility.
    densmap : bool
        Whether to use densMAP (density-preserving UMAP).

    Returns
    -------
    embedding : ndarray, shape (T, n_components)
        Low-dimensional coordinates for each sample/column in `data`.
    model : umap.UMAP
        The fitted UMAP model; use `model.transform(X_new)` to map new samples.
    """
    X = np.asarray(data)
    if X.ndim != 2:
        raise ValueError(f"`data` must be 2D, got shape {X.shape}")

    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        densmap=densmap,
        n_jobs=1,
        **umap_kwargs,
    )
    
    model.fit(X)
    
    return model


class JPCA:
    """
    jPCA implementation where both symmetric and skew-symmetric parts
    are obtained via constrained least squares.

    Input X: time x dim (already PCA-reduced, mean-centered optional).

    Model:
        dX/dt ≈ X @ M

    We compute:
        - M_full_ls : unconstrained LS solution
        - M_sym_constrained : argmin ||Xdot - X M||_F  s.t. M = M^T
        - M_skew_constrained : argmin ||Xdot - X M||_F  s.t. M = -M^T

    jPCA planes are obtained from eigen-decomposition of M_skew_constrained.

    Attributes after fit():
        M_full_ls
        M_sym_constrained
        M_skew_constrained

        evals_skew, evecs_skew  : eigenvalues/vectors of M_skew_constrained
        evals_sym, evecs_sym  : eigenvalues/vectors of M_sym_constrained
        jpc_vectors             : real jPC vectors (n x 2*k)
        kpc_vectors             : real kPC vectors (n x n)
    """

    def __init__(self, dt: float = 1.0, mean_center: bool = True):
        """
        Parameters
        ----------
        dt : float
            Time step between samples (used in finite difference derivative).
        mean_center : bool
            Whether to mean-center X along time before fitting.
        """
        self.dt = dt
        self.mean_center = mean_center

        # matrices
        self.M_full_ls = None
        self.M_sym_constrained = None
        self.M_skew_constrained = None

        # eig / jPCA
        self.evals_skew = None
        self.evecs_skew = None
        self.evals_sym = None
        self.evecs_sym = None
        self.jpc_vectors = None
        self.kpc_vectors = None


    def fit(self, X: np.ndarray):
        """
        Fit jPCA model to PCA-reduced data X (time x dim).

        Parameters
        ----------
        X : np.ndarray, shape (T, n_dims)
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D with shape (time, dims).")

        if self.mean_center:
            X = X - X.mean(axis=0, keepdims=True)

        # finite-difference derivative
        Xdot = self._compute_derivative(X)   # (T-1, n)
        Xtrim = X[:-1, :]                    # align with Xdot
        Tm1, n = Xtrim.shape

        # 1) unconstrained LS: just for reference
        M_full, *_ = np.linalg.lstsq(Xtrim, Xdot, rcond=None)
        self.M_full_ls = M_full   # shape (n, n)

        # 2) constrained LS for symmetric & skew-symmetric parts
        # self.M_sym_constrained = self._fit_sym_constrained(Xtrim, Xdot)
        self.M_skew_constrained = self._fit_skew_constrained(Xtrim, Xdot)

        # 3) eigen-decomposition of skew-constrained M (jPCA)
        evals, evecs = np.linalg.eig(self.M_skew_constrained)
        # sort by |Im(λ)| descending
        idx = np.argsort(np.abs(np.imag(evals)))[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        self.evals_skew = evals
        self.evecs_skew = evecs
        self.jpc_vectors = self._build_jpc_vectors(self.evals_skew, self.evecs_skew)
        
        self.M_sym_constrained = self._fit_sym_constrained(Xtrim, Xdot)
        
        evals, evecs = np.linalg.eigh(self.M_sym_constrained)
        idx = np.argsort(np.abs(np.imag(evals)))[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        self.evals_sym = evals
        self.evecs_sym = evecs
        
        self.kpc_vectors = self.evecs_sym


    def transform(self, X: np.ndarray, n_planes: int = 1) -> np.ndarray:
        """
        Project data onto the leading jPC planes.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_dims)
        n_planes : int
            Number of jPC planes to use (each plane = 2 dimensions).

        Returns
        -------
        X_proj : np.ndarray, shape (T, 2 * n_planes)
        """
        if self.M_skew_constrained is None or self.jpc_vectors is None:
            raise RuntimeError("Call fit(X) before transform(X).")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D (time x dims).")

        n_dims = self.M_skew_constrained.shape[0]
        if X.shape[1] != n_dims:
            raise ValueError(f"X has {X.shape[1]} dims, model expects {n_dims} dims.")

        if self.mean_center:
            X = X - X.mean(axis=0, keepdims=True)

        P = self.jpc_vectors[:, : 2 * n_planes]  # (n_dims, 2*n_planes)
        return X @ P
    
    def transform_kpc(self, X: np.ndarray, out_dim: int = 3) -> np.ndarray:
        
        if self.M_sym_constrained is None or self.kpc_vectors is None:
            raise RuntimeError("Call fit(X) before transform(X).")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D (time x dims).")

        n_dims = self.M_sym_constrained.shape[0]
        if X.shape[1] != n_dims:
            raise ValueError(f"X has {X.shape[1]} dims, model expects {n_dims} dims.")

        if self.mean_center:
            X = X - X.mean(axis=0, keepdims=True)

        P = self.kpc_vectors[:, : out_dim]  # (n_dims, out_dim)
        return X @ P

    def get_summary_matrices(self):
        """
        Convenience method: return all key matrices in a dict.
        """
        return {
            "M_full_ls": self.M_full_ls,
            "M_sym_constrained": self.M_sym_constrained,
            "M_skew_constrained": self.M_skew_constrained,
        }

    # ---------- internal helpers ----------

    def _compute_derivative(self, X: np.ndarray) -> np.ndarray:
        """
        Simple finite difference derivative along time.
        X: (T, n) -> dX/dt: (T-1, n)
        """
        return (X[1:, :] - X[:-1, :]) / self.dt


    def _fit_skew_constrained(self, X: np.ndarray, Xdot: np.ndarray) -> np.ndarray:
        """
        Solve constrained LS:
            min ||vec(Xdot) - (I ⊗ X) vec(M)||_2
            s.t. M is skew-symmetric (M = -M^T)

        Parameterization:
            vec(M) ≅ k  in R^{n(n-1)/2}  (we implement the mapping implicitly)
        """
        Tm1, n = X.shape
        assert Xdot.shape == (Tm1, n)

        # number of free params in a skew-symmetric nxn matrix
        p_skew = n * (n - 1) // 2

        # right-hand side in vec form (column-major)
        b = Xdot.reshape(-1, order="F")  # shape: (n * (T-1),)

        # --- helper: k -> M_skew (nxn) ---
        def k_to_M(k: np.ndarray) -> np.ndarray:
            M = np.zeros((n, n), dtype=X.dtype)
            idx = 0
            # for i in range(n): for j in range(i)
            for i in range(n):
                for j in range(i):
                    kij = k[idx]
                    M[i, j] += kij     # M[i,j] = +k
                    M[j, i] -= kij     # M[j,i] = -k
                    idx += 1
            return M

        # --- helper: G (nxn) -> gradient in k-space, 等价于 H^T vec(G) ---
        def G_to_k(G: np.ndarray) -> np.ndarray:
            k = np.empty(p_skew, dtype=G.dtype)
            idx = 0
            for i in range(n):
                for j in range(i):
                    # +1*G[i,j]  -1*G[j,i]
                    k[idx] = G[i, j] - G[j, i]
                    idx += 1
            return k

        # --- define LinearOperator A: R^{p_skew} -> R^{n*(T-1)} ---

        def matvec(k: np.ndarray) -> np.ndarray:
            """
            A @ k  = vec(X @ M_skew(k))
            """
            M = k_to_M(k)
            Y = X @ M                    # (T-1, n)
            return Y.reshape(-1, order="F")

        def rmatvec(v: np.ndarray) -> np.ndarray:
            """
            A.T @ v = G_to_k( X.T @ V ),
            where V reshapes v back to (T-1, n).
            """
            V = v.reshape(Tm1, n, order="F")   # (T-1, n)
            G = X.T @ V                        # (n, n)
            return G_to_k(G)

        m = n * Tm1      # rows of A
        A_op = LinearOperator(
            shape=(m, p_skew),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=X.dtype,
        )

        result = lsqr(A_op, b)
        k_opt = result[0]

        M_skew = k_to_M(k_opt)
        return M_skew

    def _fit_sym_constrained(self, X: np.ndarray, Xdot: np.ndarray) -> np.ndarray:
        """
        Solve constrained LS:
            min ||vec(Xdot) - (I ⊗ X) vec(M)||_2
            s.t. M is symmetric (M = M^T)

        Parameterization:
            M is determined by the upper triangle (including diagonal).
            We use k ∈ R^{n(n+1)/2} to encode these free parameters.
            The mapping k ↔ M is implemented implicitly (no explicit H_sym).
        """
        Tm1, n = X.shape
        assert Xdot.shape == (Tm1, n)

        # Number of free parameters for a symmetric nxn matrix
        p_sym = n * (n + 1) // 2

        # Right-hand side in vec form (column-major / Fortran order)
        b = Xdot.reshape(-1, order="F")  # shape: (n * (T-1),)

        # --- helper: k -> M_sym (nxn) ---
        def k_to_M(k: np.ndarray) -> np.ndarray:
            """
            Map parameter vector k (upper triangle including diag) to a symmetric matrix M.
            The ordering of k follows the original construction:
                for j in range(n):        # column
                    for i in range(j + 1):  # row, i <= j
                        (i, j) is the next parameter
            """
            M = np.zeros((n, n), dtype=X.dtype)
            idx = 0
            for j in range(n):          # column index
                for i in range(j + 1):  # row index, i <= j (upper triangle incl. diagonal)
                    kij = k[idx]
                    if i == j:
                        # Diagonal element
                        M[i, j] += kij
                    else:
                        # Off-diagonal mirrored to keep symmetry
                        M[i, j] += kij
                        M[j, i] += kij
                    idx += 1
            return M

        # --- helper: G (nxn) -> gradient in k-space, equivalent to H_sym^T vec(G) ---
        def G_to_k(G: np.ndarray) -> np.ndarray:
            """
            Map an nxn matrix G to the parameter-space gradient k_grad.
            This corresponds to H_sym^T vec(G) with the same ordering as k_to_M.
            For off-diagonal entries, a single parameter affects both (i, j) and (j, i),
            so the contribution is G[i, j] + G[j, i].
            """
            k_grad = np.empty(p_sym, dtype=G.dtype)
            idx = 0
            for j in range(n):          # column index
                for i in range(j + 1):  # row index, i <= j
                    if i == j:
                        # Diagonal: appears once
                        k_grad[idx] = G[i, i]
                    else:
                        # Off-diagonal: sum of symmetric positions
                        k_grad[idx] = G[i, j] + G[j, i]
                    idx += 1
            return k_grad

        # --- define LinearOperator A: R^{p_sym} -> R^{n*(T-1)} ---

        def matvec(k: np.ndarray) -> np.ndarray:
            """
            A @ k  = vec(X @ M_sym(k))
            """
            M = k_to_M(k)              # (n, n) symmetric
            Y = X @ M                  # (T-1, n)
            return Y.reshape(-1, order="F")

        def rmatvec(v: np.ndarray) -> np.ndarray:
            """
            A.T @ v = G_to_k( X.T @ V ),
            where V reshapes v back to (T-1, n).
            """
            V = v.reshape(Tm1, n, order="F")   # (T-1, n)
            G = X.T @ V                        # (n, n)
            return G_to_k(G)

        m = n * Tm1  # number of rows of A

        A_op = LinearOperator(
            shape=(m, p_sym),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=X.dtype,
        )

        # Iterative least squares solver (no explicit A, no huge SVD/QR on dense matrices)
        # You can tune atol/btol/iter_lim if needed.
        result = lsqr(A_op, b)
        k_opt = result[0]

        # Recover symmetric matrix M
        M_sym = k_to_M(k_opt)
        return M_sym


    def _build_jpc_vectors(self, evals, evecs, tol: float = 1e-9) -> np.ndarray | None:
        """
        From eigenvalues/eigenvectors of a real skew-symmetric matrix,
        build real jPC vectors (orthonormal pairs forming planes).

        For each λ = iω, eigenvector v is complex, with conjugate pair v*.
        Real and imaginary parts span the corresponding plane.

        Returns:
            jpc_vectors: np.ndarray, shape (n_dims, 2*k)
        """
        n = evecs.shape[0]
        planes = []
        used = np.zeros_like(evals, dtype=bool)

        for idx, lam in enumerate(evals):
            if used[idx]:
                continue
            if np.abs(np.imag(lam)) < tol:
                # purely real eigenvalue: no rotation
                continue

            v = evecs[:, idx]
            u1 = np.real(v)
            u2 = np.imag(v)

            # Orthonormalize u1, u2
            u1_norm = np.linalg.norm(u1)
            if u1_norm < tol:
                continue
            u1 = u1 / u1_norm

            u2 = u2 - u1 * np.dot(u1, u2)
            u2_norm = np.linalg.norm(u2)
            if u2_norm < tol:
                continue
            u2 = u2 / u2_norm

            planes.append(np.stack([u1, u2], axis=1))

            # Mark conjugate eigenvalue as used
            conj_idx = np.argmin(np.abs(evals - np.conj(lam)))
            used[idx] = True
            if np.abs(evals[conj_idx] - np.conj(lam)) < 1e-6:
                used[conj_idx] = True

        if not planes:
            return None

        return np.concatenate(planes, axis=1)



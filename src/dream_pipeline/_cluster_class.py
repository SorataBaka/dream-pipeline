from hdbscan import HDBSCAN, validity_index, approximate_predict
from umap import UMAP
from skdim.id import TwoNN
from skopt import gp_minimize
from skopt.space import Integer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.manifold import trustworthiness
import numpy as np
from collections import Counter
from scipy.stats import entropy

class DreamCluster:
    def __init__(self,
                 mode: str,
                 sample_size: int = None,
                 with_sampling: bool = False,
                 first_reductor_n_neighbor: int = 100,
                 second_reductor_n_neighbor: int = 20,
                 second_reductor_final_dimension: int = 10,
                 min_samples_min: int = 5,
                 min_samples_max: int = 500,
                 min_cluster_size_min: int = 5,
                 min_cluster_size_max: int = 500,
                 bayesian_n_calls: int = 50
                 ):
        """
        Args:
            mode: HDBSCAN optimization method to use. Can be one of ['dbcv' , 'stability'].
            sample_size: The amount of subsamples to use for clustering.
            with_sampling: Configures whether reductor clusterer transformation utilizes sampling from the original dataset.
            first_reductor_n_neighbor: Number of neighbors to use for first reductor clustering.
            second_reductor_n_neighbor: Number of neighbors to use for second reductor clustering.
            second_reductor_final_dimension: Final dimension to use for second reductor clustering.
            min_samples_min: Minimum number of samples to use for HDBSCAN optimization.
            min_samples_max: Maximum number of samples to use for HDBSCAN optimization.
            min_cluster_size_min: Minimum number of samples to use for HDBSCAN optimization.
            min_cluster_size_max: Maximum number of samples to use for HDBSCAN optimization.
            bayesian_n_calls: Number of times to run HDBSCAN optimization.
        Returns:
            None
        """
        allowed_modes = {"dbcv", "stability"}
        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode: {mode}")

        if not isinstance(with_sampling, bool):
            raise TypeError("with_sampling must be bool")
        if with_sampling and sample_size is None:
            raise ValueError("With sampling requires sample_size to be specified")

        self.mode = mode
        self.with_sampling = with_sampling
        self.sample_size = sample_size
        self.first_reductor_n_neighbor = first_reductor_n_neighbor
        self.second_reductor_n_neighbor = second_reductor_n_neighbor
        self.second_reductor_final_dimension = second_reductor_final_dimension
        self.min_samples_min = min_samples_min
        self.min_samples_max = min_samples_max
        self.min_cluster_size_min = min_cluster_size_min
        self.min_cluster_size_max = min_cluster_size_max
        self.bayesian_n_calls = bayesian_n_calls

        self.first_reductor = None
        self.second_reductor = None
        self.clusterer = None

    def predict(self, embeddings: np.ndarray):
        reduced_embeddings = self.first_reductor.transform(embeddings)
        reduced_embeddings = self.second_reductor.transform(reduced_embeddings)
        labels, probabilities = approximate_predict(self.clusterer, reduced_embeddings)
        return labels, probabilities, reduced_embeddings

    def fit(self, embeddings: np.ndarray, seed=42):
        """
        Args:
            embeddings:  a NumPy array with shape (n_samples, n_features). In this case, an output of SentenceTransformer.
            seed: random seed for reproducibility.
        Returns:
            best_min_samples: best min_samples to use with HDBSCAN from the bayesian optimization
            best_min_cluster_size: best min_cluster_size to use with HDBSCAN from the bayesian optimization
        """
        rng = np.random.default_rng(seed)
        opt_embeddings = embeddings
        if self.with_sampling and self.sample_size is not None:
            original_indices = np.arange(embeddings.shape[0])
            rng.shuffle(original_indices)
            sample_indices = original_indices[:self.sample_size]
            opt_embeddings = embeddings[sample_indices]

        print("Finding intrinsic dimension with TwoNN")
        two_nn = TwoNN()
        intrinsic_dimension = int(np.clip(np.round(two_nn.fit_transform(opt_embeddings)), 5, 50))
        print(f"Found intrinsic dimension: {intrinsic_dimension}")

        self.first_reductor = UMAP(
            n_neighbors=self.first_reductor_n_neighbor,
            n_components=intrinsic_dimension,
            metric="cosine",
            random_state=seed,
            min_dist=0.1,
            n_jobs=-1,
        )
        self.second_reductor = UMAP(
            n_neighbors=self.second_reductor_n_neighbor,
            n_components=self.second_reductor_final_dimension,
            metric="euclidean",
            random_state=seed,
            min_dist=0,
            n_jobs=-1,
        )

        print("Running first stage reduction")
        reduced_embeddings = self.first_reductor.fit_transform(opt_embeddings)
        print("Running second stage reduction")
        reduced_embeddings = self.second_reductor.fit_transform(reduced_embeddings)

        best_min_samples = best_min_cluster_size = 5
        print(f"Tuning HDBSCAN with mode {self.mode}")
        if self.mode.lower() == "dbcv":
            print("  -> Using 'dbcv' mode (Bayesian Optimization on DBCV score)")
            search_space = [
                Integer(self.min_samples_min, self.min_samples_max, name="min_samples"),
                Integer(self.min_cluster_size_min, self.min_cluster_size_max, name="min_cluster_size"),
            ]
            optimization_result = gp_minimize(
                func=lambda params: DreamCluster.__objective_for_dbcv(params, reduced_embeddings),
                dimensions=search_space,
                n_calls=self.bayesian_n_calls,
                random_state=seed,
                verbose=True,
                n_jobs=1,  # FIXED: Set to 1 for determinism
            )
            best_min_samples, best_min_cluster_size = optimization_result.x
        else:
            print(
                "  -> Using 'stability' mode (Bayesian Optimization on relative_validity)"
            )
            search_space = [
                Integer(self.min_samples_min, self.min_samples_max, name="min_samples"),
                Integer(self.min_cluster_size_min, self.min_cluster_size_max, name="min_cluster_size"),
            ]
            optimization_result = gp_minimize(
                func=lambda params: DreamCluster.__objective_for_relative_validity(
                    params, reduced_embeddings
                ),
                dimensions=search_space,
                n_calls=self.bayesian_n_calls,
                random_state=seed,
                verbose=False,
                n_jobs=1,
            )
            best_min_samples, best_min_cluster_size = optimization_result.x

        print("Transforming full dataset with trained reducers...")
        full_reduced_embeddings_1 = self.first_reductor.transform(embeddings)
        full_reduced_embeddings_final = self.second_reductor.transform(full_reduced_embeddings_1)
        self.clusterer = HDBSCAN(
            min_cluster_size=best_min_cluster_size,
            min_samples=best_min_samples,
            metric="euclidean",
            allow_single_cluster=False,
            core_dist_n_jobs=1,
            cluster_selection_method="eom",
            gen_min_span_tree=True,
            prediction_data=True
        )
        print("Fitting final clusterer on full reduced dataset...")
        self.clusterer.fit(full_reduced_embeddings_final)

        return best_min_samples, best_min_cluster_size

    @staticmethod
    def __objective_for_relative_validity(params, data):
        """
        Objective function for Bayesian optimization based on HDBSCAN
        relative validity (mean of all clusters).
        """
        try:
            min_samples = max(2, int(params[0]))
            min_cluster_size = max(2, int(params[1]))

            clusterer = HDBSCAN(
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                metric="euclidean",
                gen_min_span_tree=True,
                allow_single_cluster=False,
                core_dist_n_jobs=1,  # FIXED: Set to 1 for determinism
                cluster_selection_method="eom",
            )
            clusterer.fit(data)

            if clusterer.relative_validity_.size == 0:
                return 1.0  # Penalize "no clusters"

            # We want to maximize the mean validity, so return its negative
            return -np.mean(clusterer.relative_validity_)

        except Exception:
            # Catch any errors during clustering
            return 1.0

    @staticmethod
    def __objective_for_dbcv(params, data):
        """
        Objective function for Bayesian optimization based on the
        DBCV score.
        """
        try:
            min_samples = max(2, int(params[0]))
            min_cluster_size = max(2, int(params[1]))

            if min_samples > min_cluster_size:
                return 1.0

            clusterer = HDBSCAN(
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                metric="euclidean",
                gen_min_span_tree=True,
                cluster_selection_method="eom",
                core_dist_n_jobs=1,  # FIXED: Set to 1 for determinism
            )
            clusterer.fit(data)
            labels = clusterer.labels_

            non_noise_indices = np.where(labels != -1)[0]
            unique_labels = np.unique(labels[non_noise_indices])

            if len(unique_labels) < 2:
                return 1.0

            non_noise_data = data[non_noise_indices]
            non_noise_labels = labels[non_noise_indices]

            dbcv_score = validity_index(
                non_noise_data, non_noise_labels, metric="euclidean",
                verbose=False  # FIXED: Set to False to reduce noise
            )

            # We want to maximize DBCV, so return its negative
            return -dbcv_score  # type: ignore

        except Exception:
            # Catch any errors during DBCV calculation
            return 1.0

    @staticmethod
    def gini(array):
        """Compute Gini coefficient of a 1D array."""
        array = np.sort(array)
        n = len(array)
        cumvals = np.cumsum(array, dtype=float)
        return (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n

    @staticmethod
    def evaluate_clusters(original_embedding, embeddings, labels, n_neighbors=5):
        original_embedding = np.asarray(original_embedding, dtype=np.float64)
        embeddings = np.asarray(embeddings, dtype=np.float64)

        clustered_mask = labels != -1
        clustered_embeddings = embeddings[clustered_mask]
        clustered_labels = labels[clustered_mask]

        n_clusters = len(np.unique(clustered_labels))
        n_samples = len(clustered_labels)

        sil_score = db_score = c_score = dbcv_score = np.nan
        trust_score = continuity_score = np.nan
        imbalance_ratio = balance_entropy = balance_entropy_normalized = gini_coeff = np.nan
        cluster_sizes = {}

        try:
            trust_score = trustworthiness(
                original_embedding, embeddings, n_neighbors=n_neighbors
            )
            continuity_score = trustworthiness(
                embeddings, original_embedding, n_neighbors=n_neighbors
            )
            dbcv_score = validity_index(embeddings, labels, "euclidean")

        except Exception as e:
            print(f"Error computing trust/continuity: {e}")

        # Only compute internal clustering metrics if at least 2 clusters and 2 points
        if n_clusters >= 2 and n_samples >= 2:
            try:
                sil_score = silhouette_score(clustered_embeddings, clustered_labels)
            except Exception as e:
                print(f"Error computing silhouette_score: {e}")
            try:
                db_score = davies_bouldin_score(clustered_embeddings, clustered_labels)
            except Exception as e:
                print(f"Error computing davies_bouldin_score: {e}")
            try:
                c_score = calinski_harabasz_score(clustered_embeddings, clustered_labels)
            except Exception as e:
                print(f"Error computing calinski_harabasz_score: {e}")
            try:
                cluster_sizes = dict(Counter(clustered_labels))
                sizes = np.array(list(cluster_sizes.values()))
                imbalance_ratio = sizes.max() / sizes.min()
                proportions = sizes / sizes.sum()
                balance_entropy = entropy(proportions)
                balance_entropy_normalized = balance_entropy / np.log(len(sizes))
                gini_coeff = gini(sizes)
            except Exception as e:
                print(f"Error computing balance scores: {e}")
        else:
            print(
                f"Warning: Only {n_clusters} cluster(s) or {n_samples} sample(s). Internal metrics set to NaN."
            )

        return {
            "silhouette_score": sil_score,
            "davies_bouldin_score": db_score,
            "calinski_score": c_score,
            "dbcv_score": dbcv_score,
            "cluster_count": n_clusters,
            "trust_score": trust_score,
            "continuity_score": continuity_score,
            "imbalance_ratio": imbalance_ratio,
            "balance_entropy": balance_entropy,
            "balance_entropy_normalized": balance_entropy_normalized,
            "gini_coefficient": gini_coeff,
            "cluster_sizes": cluster_sizes,
        }








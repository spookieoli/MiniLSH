import ctypes
import numpy as np
from typing import List, Tuple, Optional


class MiniLSH:
    """
    Python wrapper for MiniLSH - Fast LSH-based vector search for RAG applications

    Usage:
        lsh = MiniLSH()
        lsh.create_collection(768, distance='cosine')
        lsh.add_vectors(embeddings, doc_ids, categories)
        results = lsh.search(query_embedding, k=5)
    """

    def __init__(self, lib_path: str = "./libragdb.so"):
        """Initialize MiniLSH and load the shared library"""
        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise RuntimeError(f"Could not load MiniLSH library from {lib_path}: {e}")

        self._setup_function_signatures()
        self._collection_created = False

    def _setup_function_signatures(self):
        """Define C function signatures for ctypes"""

        # create_collection(int dimensions, int distance_type) -> int
        self.lib.create_collection.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.create_collection.restype = ctypes.c_int

        # add_vectors_bulk(double* vectors_flat, char** payloads, char** classes, int count, int dimensions) -> int
        self.lib.add_vectors_bulk.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.add_vectors_bulk.restype = ctypes.c_int

        # search_knn(double* query, int k) -> SearchResults*
        self.lib.search_knn.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_int
        ]
        self.lib.search_knn.restype = ctypes.c_void_p

        # free_search_results(SearchResults* sr) -> void
        self.lib.free_search_results.argtypes = [ctypes.c_void_p]
        self.lib.free_search_results.restype = None

        # Define SearchResults structure
        class SearchResults(ctypes.Structure):
            _fields_ = [
                ("distances", ctypes.POINTER(ctypes.c_double)),
                ("payloads", ctypes.POINTER(ctypes.c_char_p)),
                ("classes", ctypes.POINTER(ctypes.c_char_p)),
                ("count", ctypes.c_int),
                ("buckets_checked", ctypes.c_int)
            ]

        self.SearchResults = SearchResults

    def create_collection(self, dimensions: int, distance: str = 'euclidean') -> bool:
        """
        Create a new vector collection

        Args:
            dimensions: Vector dimensionality (e.g., 768 for OpenAI embeddings)
            distance: 'euclidean' or 'cosine'

        Returns:
            True if successful, False otherwise
        """
        distance_map = {
            'euclidean': 0,  # EUCLID
            'cosine': 1  # COSINE
        }

        if distance not in distance_map:
            raise ValueError(f"Distance must be 'euclidean' or 'cosine', got '{distance}'")

        result = self.lib.create_collection(dimensions, distance_map[distance])
        self._collection_created = (result == 1)
        self._dimensions = dimensions if self._collection_created else None

        return self._collection_created

    def add_vectors(self,
                    vectors,
                    payloads: List[str],
                    classes: Optional[List[str]] = None) -> int:
        """
        Add vectors to the collection

        Args:
            vectors: 2D numpy array of shape (n_vectors, dimensions) OR list of lists
            payloads: List of string identifiers for each vector (e.g., document IDs)
            classes: Optional list of class labels for each vector

        Returns:
            Number of vectors successfully added
        """
        if not self._collection_created:
            raise RuntimeError("No collection created. Call create_collection() first.")

        # Convert list of lists to numpy array if needed
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float64)
        elif isinstance(vectors, np.ndarray):
            if vectors.dtype != np.float64:
                vectors = vectors.astype(np.float64)
        else:
            raise ValueError("Vectors must be a numpy array or list of lists")

        # Validate shape
        if vectors.ndim != 2:
            raise ValueError("Vectors must be 2D (n_vectors, dimensions)")

        n_vectors, dims = vectors.shape

        if dims != self._dimensions:
            raise ValueError(f"Vector dimensions ({dims}) don't match collection dimensions ({self._dimensions})")

        if len(payloads) != n_vectors:
            raise ValueError(f"Number of payloads ({len(payloads)}) doesn't match number of vectors ({n_vectors})")

        # Use default classes if not provided
        if classes is None:
            classes = [f"class_{i}" for i in range(n_vectors)]
        elif len(classes) != n_vectors:
            raise ValueError(f"Number of classes ({len(classes)}) doesn't match number of vectors ({n_vectors})")

        # Convert strings to C-compatible format
        payload_ptrs = (ctypes.c_char_p * n_vectors)()
        class_ptrs = (ctypes.c_char_p * n_vectors)()

        for i in range(n_vectors):
            payload_ptrs[i] = payloads[i].encode('utf-8')
            class_ptrs[i] = classes[i].encode('utf-8')

        # Flatten vectors for C function
        vectors_flat = vectors.flatten()

        # Call C function
        result = self.lib.add_vectors_bulk(
            vectors_flat,
            payload_ptrs,
            class_ptrs,
            n_vectors,
            self._dimensions
        )

        return result

    def search(self, query, k: int = 10) -> Tuple[List[float], List[str], List[str], int]:
        """
        Search for k nearest neighbors

        Args:
            query: 1D numpy array OR list of query vector
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, payloads, classes, buckets_checked)
        """
        if not self._collection_created:
            raise RuntimeError("No collection created. Call create_collection() first.")

        # Convert list to numpy array if needed
        if isinstance(query, list):
            query = np.array(query, dtype=np.float64)
        elif isinstance(query, np.ndarray):
            if query.dtype != np.float64:
                query = query.astype(np.float64)
        else:
            raise ValueError("Query must be a numpy array or list")

        # Validate query
        if query.ndim != 1:
            raise ValueError("Query must be 1D")

        if len(query) != self._dimensions:
            raise ValueError(f"Query dimensions ({len(query)}) don't match collection dimensions ({self._dimensions})")

        # Call C function
        result_ptr = self.lib.search_knn(query, k)

        if not result_ptr:
            return [], [], [], 0

        # Cast to SearchResults structure
        results = ctypes.cast(result_ptr, ctypes.POINTER(self.SearchResults)).contents

        # Extract results
        distances = []
        payloads = []
        classes = []

        for i in range(results.count):
            distances.append(results.distances[i])
            payloads.append(results.payloads[i].decode('utf-8'))
            classes.append(results.classes[i].decode('utf-8'))

        buckets_checked = results.buckets_checked

        # Free C memory
        self.lib.free_search_results(result_ptr)

        return distances, payloads, classes, buckets_checked

    def get_stats(self) -> dict:
        """Get basic statistics about the current collection"""
        if not self._collection_created:
            return {"status": "No collection created"}

        return {
            "status": "Collection active",
            "dimensions": self._dimensions
        }


# Example usage and testing
if __name__ == "__main__":
    # Create MiniLSH instance
    lsh = MiniLSH("./libragdb.so")

    # Create collection for 768-dimensional vectors with cosine similarity
    print("Creating collection...")
    success = lsh.create_collection(768, distance='cosine')
    print(f"Collection created: {success}")

    # Generate sample data (simulating document embeddings)
    print("\nGenerating sample data...")
    n_docs = 100000
    dimensions = 768

    # Random embeddings (in practice, these would come from your embedding model)
    # Test both numpy arrays and lists of lists
    embeddings_numpy = np.random.randn(n_docs, dimensions).astype(np.float64)
    embeddings_lists = embeddings_numpy.tolist()  # Convert to list of lists

    # Sample document IDs and categories
    doc_ids = [f"doc_{i:04d}" for i in range(n_docs)]
    categories = [f"category_{i % 10}" for i in range(n_docs)]

    # Test adding vectors as numpy array
    print(f"Adding {n_docs} vectors (numpy array)...")
    added = lsh.add_vectors(embeddings_numpy, doc_ids, categories)
    print(f"Added {added} vectors successfully")

    # Create new collection to test list of lists
    print("\nCreating new collection to test list of lists...")
    lsh.create_collection(768, distance='cosine')

    # Test adding vectors as list of lists
    print(f"Adding {n_docs} vectors (list of lists)...")
    added = lsh.add_vectors(embeddings_lists, doc_ids, categories)
    print(f"Added {added} vectors successfully")

    # Perform search with numpy array
    print("\nPerforming search with numpy array...")
    query_numpy = np.random.randn(dimensions).astype(np.float64)
    distances, payloads, classes, buckets_checked = lsh.search(query_numpy, k=5)

    print(f"Found {len(distances)} results (checked {buckets_checked} buckets)")
    print("Top 3 results:")
    for i, (dist, payload, cls) in enumerate(zip(distances[:3], payloads[:3], classes[:3])):
        print(f"  {i + 1}. {payload} (class: {cls}) - distance: {dist:.6f}")

    # Perform search with list
    print("\nPerforming search with list...")
    query_list = query_numpy.tolist()
    distances, payloads, classes, buckets_checked = lsh.search(query_list, k=5)

    print(f"Found {len(distances)} results (checked {buckets_checked} buckets)")
    print("Top 3 results:")
    for i, (dist, payload, cls) in enumerate(zip(distances[:3], payloads[:3], classes[:3])):
        print(f"  {i + 1}. {payload} (class: {cls}) - distance: {dist:.6f}")

    # Performance test
    print("\nPerformance test...")
    import time

    start_time = time.time()
    n_searches = 100

    for i in range(n_searches):
        # Alternate between numpy arrays and lists
        if i % 2 == 0:
            test_query = np.random.randn(dimensions).astype(np.float64)
        else:
            test_query = np.random.randn(dimensions).tolist()
        lsh.search(test_query, k=10)

    end_time = time.time()
    avg_time = (end_time - start_time) / n_searches

    print(f"Average search time: {avg_time * 1000:.3f}ms")
    print(f"Searches per second: {1 / avg_time:.0f}")
    print("(Mixed numpy arrays and lists)")

    print(f"\nStats: {lsh.get_stats()}")

    # Show usage examples
    print("\n" + "=" * 50)
    print("USAGE EXAMPLES:")
    print("=" * 50)
    print()
    print("# Both formats work for vectors:")
    print("vectors_numpy = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])")
    print("vectors_lists = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]")
    print()
    print("# Both formats work for queries:")
    print("query_numpy = np.array([1.0, 2.0, 3.0])")
    print("query_list = [1.0, 2.0, 3.0]")
    print()
    print("# All combinations work:")
    print("lsh.add_vectors(vectors_numpy, doc_ids)")
    print("lsh.add_vectors(vectors_lists, doc_ids)")
    print("lsh.search(query_numpy, k=5)")
    print("lsh.search(query_list, k=5)")
/******************************** Oliver Sharif - 07-2025 *******************************/
/* to compile: gcc -shared -fPIC -mavx2 -O3 -march=native -o libragdb.so ragdb.c -lm */
/*************************************************************************************/

#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>
#include <stdbool.h>

/* EUCLID or COSINE */
#define EUCLID 0
#define COSINE 1

// Node will hold a vector
struct Node {
    double * vector;
    char* class;
    char* payload;
};

// LSH Structures
typedef struct {
    double* hyperplane;  // Random hyperplane for hashing
} LSHHash;

typedef struct {
    struct Node** bucket;  // Nodes in this bucket
    int count;
    int capacity;
} LSHBucket;

typedef struct {
    LSHHash* hashes;      // Hash functions
    LSHBucket* buckets;   // Hash buckets
    int num_hashes;       // Number of hash functions (bits)
    int num_buckets;      // 2^num_hashes buckets
} LSHIndex;

// Collection holds vectors
struct Collection{
    struct Node ** list; // a list of all nodes to easily free them
    LSHIndex* lsh;        // LSH index for high-dimensional search
    int nodecount;
    int current_capacity;
    int dimensions;
    double (*distance_func)(double*, double*, int);
};

typedef struct {
    size_t vector_memory;
    size_t string_memory;
    size_t node_memory;
    size_t total_memory;
} MemoryInfo;

// Search result structure
typedef struct {
    struct Node* node;
    double distance;
} SearchResult;

typedef struct {
    double* distances;
    char** payloads;
    char** classes;
    int count;
    int buckets_checked;
} SearchResults;

// Function prototypes
void CreateCollection(int dimensions, int distancefunc);
void FreeCollection();
struct Node * CreateNode(double * vector, char * payload, char * class, int dimensions);
int AddVector(double * vector, char * payload, char * class);
int AddVectors(double ** vector, char ** payload, char ** class, int length);
void BuildLSHIndex();
struct Node ** IncreaseList(struct Node ** list, int oldcapacity);  // ADD THIS LINE
void DieOnError();
double euclidean_distance_avx(double* a, double* b, int dim);
double cosine_similarity_avx(double* a, double* b, int dim);
MemoryInfo GetMemoryUsage();

// LSH Functions
void InitializeLSH(int num_bits);
void FreeLSH();
unsigned int ComputeLSHHash(double* vector);
void AddToLSH(struct Node* node);
SearchResult* SearchKNearestNeighborsLSH(double* query_vector, int k, int* results_found, int* buckets_checked);

// Global collection pointer
struct Collection * col = NULL;

// Creates a new collection (automatically frees old one if exists)
void CreateCollection(int dimensions, int distancefunc){
    // Check if col is not NULL - if so, free the current collection first
    if (col != NULL){
        FreeCollection(); // Free the current collection and create another one
    }

    // Allocate new collection
    col = malloc(sizeof(struct Collection));
    col->list = malloc(1000 * sizeof(struct Node *));

    // Check for errors
    if (col == NULL || col->list == NULL){
        DieOnError();
    }

    col->distance_func = !distancefunc ? &euclidean_distance_avx : &cosine_similarity_avx;
    col->current_capacity = 1000;
    col->nodecount = 0;
    col->lsh = NULL;  // Initialize LSH to NULL
    col->dimensions = dimensions;
}

// AddVectors - bulk insert (recommended way)
int AddVectors(double ** vector, char ** payload, char ** class, int length){
    if (col == NULL || vector == NULL || payload == NULL || class == NULL || length <= 0){
        return 0;
    }

    int vectorsadded = 0;
    // Add the vectors to the collection
    for (int i = 0; i < length; i++){
        if (vector[i] != NULL && payload[i] != NULL && class[i] != NULL) {
            vectorsadded += AddVector(vector[i], payload[i], class[i]);
        }
    }
    
    // Build LSH index after adding all vectors
    if (vectorsadded > 0) {
        BuildLSHIndex();
    }
    
    return vectorsadded;
}

// AddVector - single vector insert
int AddVector(double * vector, char * payload, char * class){
    if (col == NULL){
        return 0; 
    }

    // Check if there is enough capacity
    if (col->current_capacity == col->nodecount){
        col->list = IncreaseList(col->list, col->current_capacity);
        if (col->list == NULL){
            DieOnError();
        }
    }

    // Create new node
    struct Node * node = CreateNode(vector, payload, class, col->dimensions);
    if (node == NULL){
        DieOnError();
    }

    // Add to list and increase nodecount
    col->list[col->nodecount++] = node;
    return 1;
}

// CreateNode - creates a new node with payload and class
struct Node * CreateNode(double * vector, char * payload, char * class, int dimensions){
    if (col == NULL){
        return NULL;
    }
    
    struct Node * node = malloc(sizeof(struct Node));
    if (node == NULL){
        return NULL;
    }

    // Allocate memory for strings and vector
    node->class = malloc((strlen(class)+1) * sizeof(char));
    node->payload = malloc((strlen(payload)+1) * sizeof(char));
    node->vector = malloc(dimensions * sizeof(double));
    
    // Check for allocation errors
    if (node->class == NULL || node->payload == NULL || node->vector == NULL){
        if (node->class) free(node->class);
        if (node->payload) free(node->payload);
        if (node->vector) free(node->vector);
        free(node);
        return NULL;
    } 

    // Copy the data
    strcpy(node->class, class);
    strcpy(node->payload, payload);
    node->class[strlen(class)] = '\0';
    node->payload[strlen(payload)] = '\0';
    memcpy(node->vector, vector, dimensions * sizeof(double));
    
    return node;
}

// Frees all nodes
void FreeNodes(){
    if (col == NULL || col->list == NULL) return;
    
    for (int i = 0; i < col->nodecount; i++){
        if (col->list[i] != NULL) {
            if (col->list[i]->vector) free(col->list[i]->vector);
            if (col->list[i]->class) free(col->list[i]->class);
            if (col->list[i]->payload) free(col->list[i]->payload);
            free(col->list[i]);
            col->list[i] = NULL;
        }
    }
    free(col->list);
    col->list = NULL;
}

// Frees the entire collection
void FreeCollection(){
    if (col == NULL) return;
    
    FreeLSH();  // Free LSH structures first
    FreeNodes(); // Then free all nodes
    free(col);
    col = NULL;  // Prevent dangling pointer
}

// Reallocate list to get more memory
struct Node ** IncreaseList(struct Node ** list, int oldcapacity){
    struct Node ** new_list = realloc(list, (oldcapacity + 1000) * sizeof(struct Node *));
    if (new_list != NULL){
        col->current_capacity = oldcapacity + 1000;
        return new_list;
    } else {
        return list; // Return original list if realloc failed
    }
}

// Initialize LSH index
void InitializeLSH(int num_bits) {
    if (col == NULL) return;
    
    col->lsh = malloc(sizeof(LSHIndex));
    if (!col->lsh) return;
    
    col->lsh->num_hashes = num_bits;
    col->lsh->num_buckets = 1 << num_bits;  // 2^num_bits
    
    // Create hash functions (random hyperplanes)
    col->lsh->hashes = malloc(num_bits * sizeof(LSHHash));
    if (!col->lsh->hashes) {
        free(col->lsh);
        col->lsh = NULL;
        return;
    }
    
    srand(42);  // Fixed seed for reproducible results
    
    for (int i = 0; i < num_bits; i++) {
        col->lsh->hashes[i].hyperplane = malloc(col->dimensions * sizeof(double));
        if (!col->lsh->hashes[i].hyperplane) {
            // Cleanup on error
            for (int j = 0; j < i; j++) {
                free(col->lsh->hashes[j].hyperplane);
            }
            free(col->lsh->hashes);
            free(col->lsh);
            col->lsh = NULL;
            return;
        }
        
        // Generate random hyperplane
        for (int j = 0; j < col->dimensions; j++) {
            col->lsh->hashes[i].hyperplane[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    
    // Initialize buckets
    col->lsh->buckets = malloc(col->lsh->num_buckets * sizeof(LSHBucket));
    if (!col->lsh->buckets) {
        for (int i = 0; i < num_bits; i++) {
            free(col->lsh->hashes[i].hyperplane);
        }
        free(col->lsh->hashes);
        free(col->lsh);
        col->lsh = NULL;
        return;
    }
    
    for (int i = 0; i < col->lsh->num_buckets; i++) {
        col->lsh->buckets[i].bucket = malloc(100 * sizeof(struct Node*));
        col->lsh->buckets[i].count = 0;
        col->lsh->buckets[i].capacity = 100;
        
        if (!col->lsh->buckets[i].bucket) {
            // Cleanup on error
            for (int j = 0; j < i; j++) {
                free(col->lsh->buckets[j].bucket);
            }
            free(col->lsh->buckets);
            for (int j = 0; j < num_bits; j++) {
                free(col->lsh->hashes[j].hyperplane);
            }
            free(col->lsh->hashes);
            free(col->lsh);
            col->lsh = NULL;
            return;
        }
    }
}

// Free LSH structures
void FreeLSH() {
    if (col == NULL || col->lsh == NULL) return;
    
    if (col->lsh->hashes) {
        for (int i = 0; i < col->lsh->num_hashes; i++) {
            if (col->lsh->hashes[i].hyperplane) {
                free(col->lsh->hashes[i].hyperplane);
            }
        }
        free(col->lsh->hashes);
    }
    
    if (col->lsh->buckets) {
        for (int i = 0; i < col->lsh->num_buckets; i++) {
            if (col->lsh->buckets[i].bucket) {
                free(col->lsh->buckets[i].bucket);
            }
        }
        free(col->lsh->buckets);
    }
    
    free(col->lsh);
    col->lsh = NULL;
}

// Compute LSH hash for a vector using AVX optimization
unsigned int ComputeLSHHash(double* vector) {
    if (col->lsh == NULL) return 0;
    
    unsigned int hash = 0;
    
    for (int i = 0; i < col->lsh->num_hashes; i++) {
        double dot_product = 0.0;
        
        // Use AVX for dot product computation
        __m256d sum = _mm256_setzero_pd();
        int j;
        
        for (j = 0; j <= col->dimensions - 4; j += 4) {
            __m256d va = _mm256_loadu_pd(&vector[j]);
            __m256d vb = _mm256_loadu_pd(&col->lsh->hashes[i].hyperplane[j]);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(va, vb));
        }
        
        // Sum the elements in the __m256d register
        __m256d temp = _mm256_hadd_pd(sum, sum);
        temp = _mm256_hadd_pd(temp, temp);
        __m128d sum_high = _mm256_extractf128_pd(temp, 1);
        __m128d result = _mm_add_pd(_mm256_castpd256_pd128(temp), sum_high);
        dot_product = _mm_cvtsd_f64(_mm_hadd_pd(result, result));
        
        // Handle remaining elements
        for (; j < col->dimensions; j++) {
            dot_product += vector[j] * col->lsh->hashes[i].hyperplane[j];
        }
        
        // Set bit if positive
        if (dot_product >= 0.0) {
            hash |= (1U << i);
        }
    }
    
    return hash;
}

// Add node to LSH index
void AddToLSH(struct Node* node) {
    if (col->lsh == NULL || node == NULL) return;
    
    unsigned int hash = ComputeLSHHash(node->vector);
    LSHBucket* bucket = &col->lsh->buckets[hash];
    
    // Expand bucket if needed
    if (bucket->count >= bucket->capacity) {
        bucket->capacity *= 2;
        struct Node** new_bucket = realloc(bucket->bucket, bucket->capacity * sizeof(struct Node*));
        if (new_bucket) {
            bucket->bucket = new_bucket;
        } else {
            return;  // Skip this node on allocation failure
        }
    }
    
    bucket->bucket[bucket->count++] = node;
}

// Build LSH index for all vectors in collection
void BuildLSHIndex(){
    if (col == NULL || col->list == NULL || col->nodecount == 0){
        return;
    }

    // Initialize LSH if not done yet (choose bits based on dataset size)
    if (col->lsh == NULL) {
        int num_bits = 16;  // 2^16 = 65k buckets
        if (col->nodecount < 10000) num_bits = 12;       // 4k buckets
        else if (col->nodecount < 50000) num_bits = 14;  // 16k buckets
        else if (col->nodecount > 200000) num_bits = 18; // 256k buckets
        
        InitializeLSH(num_bits);
        
        if (col->lsh == NULL) {
            printf("Warning: LSH initialization failed\n");
            return;
        }
    }
    
    // Clear existing LSH buckets and rebuild
    for (int i = 0; i < col->lsh->num_buckets; i++) {
        col->lsh->buckets[i].count = 0;
    }
    
    // Add all nodes to LSH
    for (int i = 0; i < col->nodecount; i++) {
        AddToLSH(col->list[i]);
    }
}

// LSH Search function
SearchResult* SearchKNearestNeighborsLSH(double* query_vector, int k, int* results_found, int* buckets_checked) {
    if (col == NULL || col->lsh == NULL || query_vector == NULL || k <= 0) {
        *results_found = 0;
        *buckets_checked = 0;
        return NULL;
    }
    
    // Compute hash for query
    unsigned int query_hash = ComputeLSHHash(query_vector);
    
    // Create candidate list
    struct Node** candidates = malloc(col->nodecount * sizeof(struct Node*));
    if (!candidates) {
        *results_found = 0;
        *buckets_checked = 0;
        return NULL;
    }
    
    int num_candidates = 0;
    *buckets_checked = 0;
    
    // Check buckets in order of Hamming distance to query_hash
    for (int hamming_dist = 0; hamming_dist <= col->lsh->num_hashes && num_candidates < k * 20; hamming_dist++) {
        
        // Generate all hashes at this Hamming distance
        for (unsigned int bucket_hash = 0; bucket_hash < col->lsh->num_buckets; bucket_hash++) {
            
            // Count different bits between query_hash and bucket_hash
            unsigned int xor_result = query_hash ^ bucket_hash;
            int bits_different = __builtin_popcount(xor_result);
            
            if (bits_different == hamming_dist) {
                (*buckets_checked)++;
                LSHBucket* bucket = &col->lsh->buckets[bucket_hash];
                
                // Add all nodes from this bucket (avoid duplicates)
                for (int i = 0; i < bucket->count; i++) {
                    // Simple duplicate check
                    bool is_duplicate = false;
                    for (int j = 0; j < num_candidates; j++) {
                        if (candidates[j] == bucket->bucket[i]) {
                            is_duplicate = true;
                            break;
                        }
                    }
                    
                    if (!is_duplicate && num_candidates < col->nodecount) {
                        candidates[num_candidates++] = bucket->bucket[i];
                    }
                }
            }
        }
    }
    
    if (num_candidates == 0) {
        free(candidates);
        *results_found = 0;
        return NULL;
    }
    
    // Calculate distances and find k best
    SearchResult* all_results = malloc(num_candidates * sizeof(SearchResult));
    if (!all_results) {
        free(candidates);
        *results_found = 0;
        return NULL;
    }
    
    for (int i = 0; i < num_candidates; i++) {
        all_results[i].node = candidates[i];
        all_results[i].distance = col->distance_func(query_vector, candidates[i]->vector, col->dimensions);
    }
    
    // Sort candidates by distance
    for (int i = 0; i < num_candidates - 1; i++) {
        for (int j = i + 1; j < num_candidates; j++) {
            if (all_results[i].distance > all_results[j].distance) {
                SearchResult temp = all_results[i];
                all_results[i] = all_results[j];
                all_results[j] = temp;
            }
        }
    }
    
    // Return top k with bounds checking
    int actual_k = (k < num_candidates) ? k : num_candidates;
    if (actual_k <= 0 || actual_k > 10000) {  // Safety bounds
        free(candidates);
        free(all_results);
        *results_found = 0;
        return NULL;
    }
    
    SearchResult* results = malloc(actual_k * sizeof(SearchResult));
    if (!results) {
        free(candidates);
        free(all_results);
        *results_found = 0;
        return NULL;
    }
    
    for (int i = 0; i < actual_k; i++) {
        results[i] = all_results[i];
    }
    
    *results_found = actual_k;
    
    free(candidates);
    free(all_results);
    return results;
}

MemoryInfo GetMemoryUsage() {
    MemoryInfo info = {0, 0, 0, 0};
    
    if (col == NULL) return info;
    
    info.vector_memory = col->nodecount * col->dimensions * sizeof(double);
    info.node_memory = col->nodecount * sizeof(struct Node);
    
    // Estimate string memory
    info.string_memory = col->nodecount * 70; // Rough estimate
    
    // Add LSH memory
    if (col->lsh != NULL) {
        info.vector_memory += col->lsh->num_hashes * col->dimensions * sizeof(double); // hyperplanes
        info.node_memory += col->lsh->num_buckets * sizeof(LSHBucket); // bucket structures
    }
    
    info.total_memory = info.vector_memory + info.string_memory + info.node_memory;
    
    return info;
}

// euclidean_distance_avx will use intrinsics to calculate euclidean distance
double euclidean_distance_avx(double* a, double* b, int dim){
    __m256d sum = _mm256_setzero_pd();
    int i;

    // Process 16 elements at a time
    for (i = 0; i <= dim - 16; i += 16) {
        __m256d va1 = _mm256_loadu_pd(&a[i]);
        __m256d vb1 = _mm256_loadu_pd(&b[i]);
        __m256d diff1 = _mm256_sub_pd(va1, vb1);
        __m256d sq1 = _mm256_mul_pd(diff1, diff1);

        __m256d va2 = _mm256_loadu_pd(&a[i + 4]);
        __m256d vb2 = _mm256_loadu_pd(&b[i + 4]);
        __m256d diff2 = _mm256_sub_pd(va2, vb2);
        __m256d sq2 = _mm256_mul_pd(diff2, diff2);

        __m256d va3 = _mm256_loadu_pd(&a[i + 8]);
        __m256d vb3 = _mm256_loadu_pd(&b[i + 8]);
        __m256d diff3 = _mm256_sub_pd(va3, vb3);
        __m256d sq3 = _mm256_mul_pd(diff3, diff3);

        __m256d va4 = _mm256_loadu_pd(&a[i + 12]);
        __m256d vb4 = _mm256_loadu_pd(&b[i + 12]);
        __m256d diff4 = _mm256_sub_pd(va4, vb4);
        __m256d sq4 = _mm256_mul_pd(diff4, diff4);

        sum = _mm256_add_pd(sum, sq1);
        sum = _mm256_add_pd(sum, sq2);
        sum = _mm256_add_pd(sum, sq3);
        sum = _mm256_add_pd(sum, sq4);
    }

    // Handle remaining elements in chunks of 4
    for (; i <= dim - 4; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d diff = _mm256_sub_pd(va, vb);
        __m256d sq = _mm256_mul_pd(diff, diff);
        sum = _mm256_add_pd(sum, sq);
    }

    // Sum the elements in the __m256d register
    __m256d temp = _mm256_hadd_pd(sum, sum);
    temp = _mm256_hadd_pd(temp, temp);
    __m128d sum_high = _mm256_extractf128_pd(temp, 1);
    __m128d result = _mm_add_pd(_mm256_castpd256_pd128(temp), sum_high);
    double final_sum = _mm_cvtsd_f64(_mm_hadd_pd(result, result));

    // Handle remaining elements one by one
    for (; i < dim; i++) {
        double diff = a[i] - b[i];
        final_sum += diff * diff;
    }

    return sqrt(final_sum);
}

// cosine_similarity_avx will use intrinsics to calculate cosine similarity
double cosine_similarity_avx(double * a, double * b, int dim){
    __m256d sum_a = _mm256_setzero_pd();
    __m256d sum_b = _mm256_setzero_pd();
    __m256d sum_ab = _mm256_setzero_pd();
    int i;

    // Process 16 elements at a time
    for (i = 0; i <= dim - 16; i += 16) {
        __m256d va1 = _mm256_loadu_pd(&a[i]);
        __m256d vb1 = _mm256_loadu_pd(&b[i]);
        sum_ab = _mm256_add_pd(sum_ab, _mm256_mul_pd(va1, vb1));
        sum_a = _mm256_add_pd(sum_a, _mm256_mul_pd(va1, va1));
        sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(vb1, vb1));

        __m256d va2 = _mm256_loadu_pd(&a[i + 4]);
        __m256d vb2 = _mm256_loadu_pd(&b[i + 4]);
        sum_ab = _mm256_add_pd(sum_ab, _mm256_mul_pd(va2, vb2));
        sum_a = _mm256_add_pd(sum_a, _mm256_mul_pd(va2, va2));
        sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(vb2, vb2));

        __m256d va3 = _mm256_loadu_pd(&a[i + 8]);
        __m256d vb3 = _mm256_loadu_pd(&b[i + 8]);
        sum_ab = _mm256_add_pd(sum_ab, _mm256_mul_pd(va3, vb3));
        sum_a = _mm256_add_pd(sum_a, _mm256_mul_pd(va3, va3));
        sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(vb3, vb3));

        __m256d va4 = _mm256_loadu_pd(&a[i + 12]);
        __m256d vb4 = _mm256_loadu_pd(&b[i + 12]);
        sum_ab = _mm256_add_pd(sum_ab, _mm256_mul_pd(va4, vb4));
        sum_a = _mm256_add_pd(sum_a, _mm256_mul_pd(va4, va4));
        sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(vb4, vb4));
    }

    // Handle remaining elements in chunks of 4
    for (; i <= dim - 4; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        sum_ab = _mm256_add_pd(sum_ab, _mm256_mul_pd(va, vb));
        sum_a = _mm256_add_pd(sum_a, _mm256_mul_pd(va, va));
        sum_b = _mm256_add_pd(sum_b, _mm256_mul_pd(vb, vb));
    }

    // Sum the elements in the __m256d registers
    __m256d temp_ab = _mm256_add_pd(sum_ab, _mm256_permute2f128_pd(sum_ab, sum_ab, 1));
    temp_ab = _mm256_add_pd(temp_ab, _mm256_permute_pd(temp_ab, 0x5));
    double final_sum_ab = _mm_cvtsd_f64(_mm256_castpd256_pd128(temp_ab)) + _mm_cvtsd_f64(_mm256_extractf128_pd(temp_ab, 1));

    __m256d temp_a = _mm256_add_pd(sum_a, _mm256_permute2f128_pd(sum_a, sum_a, 1));
    temp_a = _mm256_add_pd(temp_a, _mm256_permute_pd(temp_a, 0x5));
    double final_sum_a = _mm_cvtsd_f64(_mm256_castpd256_pd128(temp_a)) + _mm_cvtsd_f64(_mm256_extractf128_pd(temp_a, 1));

    __m256d temp_b = _mm256_add_pd(sum_b, _mm256_permute2f128_pd(sum_b, sum_b, 1));
    temp_b = _mm256_add_pd(temp_b, _mm256_permute_pd(temp_b, 0x5));
    double final_sum_b = _mm_cvtsd_f64(_mm256_castpd256_pd128(temp_b)) + _mm_cvtsd_f64(_mm256_extractf128_pd(temp_b, 1));

    // Handle remaining elements one by one
    for (; i < dim; i++) {
        double va = a[i];
        double vb = b[i];
        final_sum_ab += va * vb;
        final_sum_a += va * va;
        final_sum_b += vb * vb;
    }

    return 1.0 - (final_sum_ab / (sqrt(final_sum_a) * sqrt(final_sum_b)));
}

// DieOnError - exits program on critical error
void DieOnError(){
    exit(1);
}

// Collection
int create_collection(int dimensions, int distance_type) {
    CreateCollection(dimensions, distance_type);
    return col != NULL ? 1 : 0;
}

// Bulk insert 
int add_vectors_bulk(double* vectors_flat, char** payloads, char** classes, int count, int dimensions) {
    // Konvertiere flaches Array zu double**
    double** vectors = malloc(count * sizeof(double*));
    for (int i = 0; i < count; i++) {
        vectors[i] = &vectors_flat[i * dimensions];
    }
    
    int result = AddVectors(vectors, payloads, classes, count);
    free(vectors);
    return result;
}

// Search wrapper
SearchResults* search_knn(double* query, int k) {
    int results_found, buckets_checked;
    SearchResult* results = SearchKNearestNeighborsLSH(query, k, &results_found, &buckets_checked);
    
    if (!results) return NULL;
    
    SearchResults* sr = malloc(sizeof(SearchResults));
    sr->distances = malloc(results_found * sizeof(double));
    sr->payloads = malloc(results_found * sizeof(char*));
    sr->classes = malloc(results_found * sizeof(char*));
    sr->count = results_found;
    sr->buckets_checked = buckets_checked;
    
    for (int i = 0; i < results_found; i++) {
        sr->distances[i] = results[i].distance;
        sr->payloads[i] = results[i].node->payload;
        sr->classes[i] = results[i].node->class;
    }
    
    free(results);
    return sr;
}

// Results cleanup
void free_search_results(SearchResults* sr) {
    if (sr) {
        free(sr->distances);
        free(sr->payloads);
        free(sr->classes);
        free(sr);
    }
}
#include <cuvs/cluster/kmeans.h>
#include <cuvs/preprocessing/pca.h>
#include <stddef.h>

_Static_assert(sizeof(struct cuvsKMeansParams_v2) == 72, "unexpected cuvsKMeansParams_v2 size");
_Static_assert(_Alignof(struct cuvsKMeansParams_v2) == 8, "unexpected cuvsKMeansParams_v2 alignment");
_Static_assert(offsetof(struct cuvsKMeansParams_v2, metric) == 0, "unexpected KMeans metric offset");
_Static_assert(offsetof(struct cuvsKMeansParams_v2, hierarchical) == 48, "unexpected KMeans hierarchical offset");
_Static_assert(offsetof(struct cuvsKMeansParams_v2, streaming_batch_size) == 56,
               "unexpected KMeans streaming_batch_size offset");
_Static_assert(offsetof(struct cuvsKMeansParams_v2, init_size) == 64, "unexpected KMeans init_size offset");
_Static_assert(sizeof(struct cuvsPcaParams) == 20, "unexpected cuvsPcaParams size");
_Static_assert(_Alignof(struct cuvsPcaParams) == 4, "unexpected cuvsPcaParams alignment");

size_t cuvs_nexus_sizeof_kmeans_params_v2(void) { return sizeof(struct cuvsKMeansParams_v2); }
size_t cuvs_nexus_alignof_kmeans_params_v2(void) { return _Alignof(struct cuvsKMeansParams_v2); }
size_t cuvs_nexus_offsetof_kmeans_streaming_batch_size_v2(void)
{
  return offsetof(struct cuvsKMeansParams_v2, streaming_batch_size);
}
size_t cuvs_nexus_sizeof_pca_params(void) { return sizeof(struct cuvsPcaParams); }
size_t cuvs_nexus_alignof_pca_params(void) { return _Alignof(struct cuvsPcaParams); }

#include <cuvs/cluster/kmeans.h>
#include <cuvs/preprocessing/pca.h>
#include <stddef.h>

_Static_assert(sizeof(struct cuvsKMeansParams) == 72, "unexpected cuvsKMeansParams size");
_Static_assert(_Alignof(struct cuvsKMeansParams) == 8, "unexpected cuvsKMeansParams alignment");
_Static_assert(offsetof(struct cuvsKMeansParams, metric) == 0, "unexpected KMeans metric offset");
_Static_assert(offsetof(struct cuvsKMeansParams, hierarchical) == 48, "unexpected KMeans hierarchical offset");
_Static_assert(offsetof(struct cuvsKMeansParams, device_buffer_samples) == 56,
               "unexpected KMeans device_buffer_samples offset");
_Static_assert(offsetof(struct cuvsKMeansParams, init_size) == 64, "unexpected KMeans init_size offset");
_Static_assert(sizeof(struct cuvsPcaParams) == 20, "unexpected cuvsPcaParams size");
_Static_assert(_Alignof(struct cuvsPcaParams) == 4, "unexpected cuvsPcaParams alignment");

size_t cuvs_nexus_sizeof_kmeans_params(void) { return sizeof(struct cuvsKMeansParams); }
size_t cuvs_nexus_alignof_kmeans_params(void) { return _Alignof(struct cuvsKMeansParams); }
size_t cuvs_nexus_offsetof_kmeans_device_buffer_samples(void)
{
  return offsetof(struct cuvsKMeansParams, device_buffer_samples);
}
size_t cuvs_nexus_sizeof_pca_params(void) { return sizeof(struct cuvsPcaParams); }
size_t cuvs_nexus_alignof_pca_params(void) { return _Alignof(struct cuvsPcaParams); }

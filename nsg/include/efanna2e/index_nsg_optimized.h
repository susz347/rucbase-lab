#ifndef EFANNA2E_INDEX_NSG_OPTIMIZED_H
#define EFANNA2E_INDEX_NSG_OPTIMIZED_H

#include "index_nsg.h"

namespace efanna2e {

class IndexNSGOptimized : public IndexNSG {
 public:
  explicit IndexNSGOptimized(const size_t dimension, const size_t n, Metric m,
                             Index *initializer)
      : IndexNSG(dimension, n, m, initializer) {
    visited_versions_.resize(n, 0);
    current_version_ = 0;
  }

  // Version-array based search: avoids O(n) visited-flag reset
  void SearchWithVersionArray(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices);

  // Enhanced prefetch search with version array
  void SearchWithOptGraphV2(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices);

 protected:
  std::vector<unsigned> visited_versions_;
  unsigned current_version_;
};

}

#endif // EFANNA2E_INDEX_NSG_OPTIMIZED_H

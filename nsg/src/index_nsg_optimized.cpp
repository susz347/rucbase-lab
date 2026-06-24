#include "efanna2e/index_nsg_optimized.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <algorithm>
#include <chrono>

namespace efanna2e {

void IndexNSGOptimized::SearchWithVersionArray(
    const float *query,
    size_t K,
    const Parameters &parameters,
    unsigned *indices)
{
  const unsigned L = parameters.Get<unsigned>("L_search");

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);

  // Version array: increment global version instead of resetting flags
  current_version_++;
  unsigned cur_ver = current_version_;

  unsigned tmp_l = 0;
  for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
    init_ids[tmp_l] = final_graph_[ep_][tmp_l];
    visited_versions_[init_ids[tmp_l]] = cur_ver;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (visited_versions_[id] == cur_ver) continue;
    visited_versions_[id] = cur_ver;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float dist = distance_->compare(data_ + dimension_ * id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (visited_versions_[id] == cur_ver) continue;
        visited_versions_[id] = cur_ver;

        float dist = distance_->compare(query, data_ + dimension_ * id,
                                        (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSGOptimized::SearchWithOptGraphV2(
    const float *query,
    size_t K,
    const Parameters &parameters,
    unsigned *indices)
{
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);

  // Version array: avoids O(n) flag reset
  current_version_++;
  unsigned cur_ver = current_version_;

  unsigned tmp_l = 0;
  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
  unsigned MaxM_ep = *neighbors;
  neighbors++;

  for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
    init_ids[tmp_l] = neighbors[tmp_l];
    visited_versions_[init_ids[tmp_l]] = cur_ver;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (visited_versions_[id] == cur_ver) continue;
    visited_versions_[id] = cur_ver;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  // Prefetch all init nodes
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float *x = (float *)(opt_graph_ + node_size * id);
    float norm_x = *x;
    x++;
    float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      // Prefetch node metadata and neighbors
      _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;

      // Prefetch all neighbor data (two hops ahead)
      for (unsigned m = 0; m < MaxM && m < 8; ++m) {
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
      }
      for (unsigned m = 8; m < MaxM; ++m) {
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T1);
      }

      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (visited_versions_[id] == cur_ver) continue;
        visited_versions_[id] = cur_ver;

        float *data = (float *)(opt_graph_ + node_size * id);
        float norm = *data;
        data++;
        float dist = dist_fast->compare(query, data, norm, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

}

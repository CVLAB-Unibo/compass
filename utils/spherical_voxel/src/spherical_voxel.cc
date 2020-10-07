#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

std::vector<std::vector<std::vector<float> > > compute(const std::vector<std::vector<float> >& pts_on_s2,
                                                       const std::vector<float>& pts_norm,
                                                       int size_bandwidth,
                                                       int size_radial_divisions,
                                                       float radius_support,
                                                       const std::vector<float>& daas_weights) {

    const float interval = radius_support / (size_radial_divisions);
    std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > grids;
    std::vector<std::vector<std::vector<float> > > features;

    grids.resize(size_radial_divisions);
    features.resize(size_radial_divisions);

    for (auto &beta: grids) {
        beta.resize(2 * size_bandwidth);
        for (auto &alpha: beta) {
            alpha.resize(2 * size_bandwidth);
        }
    }

    for (auto &beta: features) {
        beta.resize(2 * size_bandwidth);
        for (auto &alpha: beta) {
            alpha.resize(2 * size_bandwidth, 0);
        }
    }

    for (size_t i = 0; i < pts_on_s2.size(); i++) {
        int r_idx = int(pts_norm[i] / interval);
        if (r_idx > size_radial_divisions - 1) r_idx = size_radial_divisions - 1;

        int beta_idx = int(pts_on_s2[i][0] + 0.5f);
        if (beta_idx > 2 * size_bandwidth - 1) beta_idx = 2 * size_bandwidth - 1;

        int alpha_idx = int(pts_on_s2[i][1] + 0.5f);
        if (alpha_idx > 2 * size_bandwidth - 1) alpha_idx = 2 * size_bandwidth - 1;

        grids[r_idx][beta_idx][alpha_idx].emplace_back(std::vector<float>{pts_norm[i], pts_on_s2[i][0], pts_on_s2[i][1]});
    }

    for (size_t i = 0; i < size_radial_divisions; i++) {
        for (size_t j = 0; j < 2 * size_bandwidth; j++) {
            for (size_t k = 0; k < 2 * size_bandwidth; k++) {
                const float left = std::max(0.f, k - 0.5f / daas_weights[j]);
                const float right = std::min(2.f * size_bandwidth, k + 0.5f / daas_weights[j]);
                float sum = 0.f;
                int cnt = 0;

                for (int m = int(left + 0.5f); m < int(right + 0.5f); m++) {
                    for (int n = 0; n < grids[i][j][m].size(); n++) {
                        if (grids[i][j][m][n][2] > left && grids[i][j][m][n][2] < right) {
                            sum += 1.f - std::abs(grids[i][j][m][n][0] / interval - (i + 1));
                            cnt++;
                        }
                    }

                    if (i < size_radial_divisions - 1) {
                        for (int n = 0; n < grids[i + 1][j][m].size(); n++) {
                            if (grids[i + 1][j][m][n][2] > left && grids[i + 1][j][m][n][2] < right) {
                                sum += 1.f - std::abs(grids[i + 1][j][m][n][0] / interval - (i + 1));
                                cnt++;
                            }
                        }
                    }
                }

                if (cnt > 0) {
                    features[i][j][k] = sum / cnt;
                }
            }
        }
    }
    return features;
}

PYBIND11_MODULE(spherical_voxel, m) {
    m.doc() = "pybind11 example plugin";
    m.def("compute", &compute, "Compute Spherical Voxelization.",
    py::arg("pts_on_s2"),
    py::arg("pts_norm"),
    py::arg("size_bandwidth"),
    py::arg("size_radial_divisions"),
    py::arg("radius_support"),
    py::arg("daas_weights"));
}

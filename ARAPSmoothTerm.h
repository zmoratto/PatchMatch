#ifndef __VW_STEREO_ARAPSMOOTHTERM_H__
#define __VW_STEREO_ARAPSMOOTHTERM_H__

#include <ARAPDataTerm.h>
#include <Eigen/SparseCore>

namespace vw {
  namespace stereo {
    // Weight numbers correspond to the 4 types of laplacian
    // operators.
    void generate_weight1(ImageView<float> const& a,
                          double gamma,
                          ImageView<float> & weight);
    void generate_weight2(ImageView<float> const& a,
                          double gamma,
                          ImageView<float> & weight);
    void generate_weight3(ImageView<float> const& a,
                          double gamma,
                          ImageView<float> & weight);
    void generate_weight4(ImageView<float> const& a,
                          double gamma,
                          ImageView<float> & weight);

    void generate_laplacian1(ImageView<float> const& weight,
                             Eigen::SparseMatrix<float> & sparse);
    void generate_laplacian2(ImageView<float> const& weight,
                             Eigen::SparseMatrix<float> & sparse);
    void generate_laplacian3(ImageView<float> const& weight,
                             Eigen::SparseMatrix<float> & sparse);
    void generate_laplacian4(ImageView<float> const& weight,
                             Eigen::SparseMatrix<float> & sparse);
  }
}

#endif // __VW_STEREO_ARAPSMOOTHTERM_H__

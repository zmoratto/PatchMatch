#include <ARAPSmoothTerm.h>

#include <vw/Image/Filter.h>
#include <vw/Image/ImageView.h>

#include <Eigen/Sparse>

void vw::stereo::generate_weight1(ImageView<float> const& a,
                                  double gamma,
                                  ImageView<float> & weight) {
  ImageView<float> kernel(3,1);
  kernel(0,0)=1; kernel(1,0)=-2;  kernel(2,0)=1;
  weight =
    exp(-convolution_filter(a, kernel, 1, 0, ConstantEdgeExtension())/gamma);
}
void vw::stereo::generate_weight2(ImageView<float> const& a,
                                  double gamma,
                                  ImageView<float> & weight) {
  ImageView<float> kernel(1,3);
  kernel(0,0)=1; kernel(0,1)=-2;  kernel(0,2)=1;
  weight =
    exp(-convolution_filter(a, kernel, 0, 1, ConstantEdgeExtension())/gamma);
}
void vw::stereo::generate_weight3(ImageView<float> const& a,
                                  double gamma,
                                  ImageView<float> & weight) {
  ImageView<float> kernel(3,3);
  kernel(0,0)=1; kernel(1,0)=0 ;  kernel(2,0)=0;
  kernel(0,1)=0; kernel(1,1)=-2;  kernel(2,1)=0;
  kernel(0,2)=0; kernel(1,2)=0 ;  kernel(2,2)=1;
  weight =
    exp(-convolution_filter(a, kernel, 1, 1, ConstantEdgeExtension())/gamma);
}
void vw::stereo::generate_weight4(ImageView<float> const& a,
                                  double gamma,
                                  ImageView<float> & weight) {
  ImageView<float> kernel(3,3);
  kernel(0,0)=0; kernel(1,0)=0 ;  kernel(2,0)=1;
  kernel(0,1)=0; kernel(1,1)=-2;  kernel(2,1)=0;
  kernel(0,2)=1; kernel(1,2)=0 ;  kernel(2,2)=0;
  weight =
    exp(-convolution_filter(a, kernel, 1, 1, ConstantEdgeExtension())/gamma);
}

void vw::stereo::generate_laplacian1(ImageView<float> const& weight,
                                     Eigen::SparseMatrix<float> & sparse) {
  // These matrices are column major
  sparse.reserve(Eigen::VectorXf::Constant(weight.cols() * weight.rows(), 3));
  int k = 0;
  for (int j = 0; j < weight.rows(); j++ ) {
    for (int i = 0; i < weight.cols(); i++) {
      sparse.insert(k, k) = 2 * weight(i,j);
      if (i > 0) {
        sparse.insert(k, k - 1) = -weight(i,j);
      }
      if (i < weight.cols()-1) {
        sparse.insert(k, k + 1) = -weight(i,j);
      }
      k++;
    }
  }
}
void vw::stereo::generate_laplacian2(ImageView<float> const& weight,
                                     Eigen::SparseMatrix<float> & sparse) {
  // These matrices are column major
  sparse.reserve(Eigen::VectorXf::Constant(weight.cols() * weight.rows(), 3));
  int k = 0;
  for (int j = 0; j < weight.rows(); j++ ) {
    for (int i = 0; i < weight.cols(); i++) {
      sparse.insert(k,k) = 2 * weight(i,j);
      if (j > 0) {
        sparse.insert(k, k-weight.cols()) = -weight(i,j);
      }
      if (j < weight.rows()-1) {
        sparse.insert(k, k+weight.cols()) = -weight(i,j);
      }
      k++;
    }
  }
}
void vw::stereo::generate_laplacian3(ImageView<float> const& weight,
                                     Eigen::SparseMatrix<float> & sparse) {
  // These matrices are column major
  sparse.reserve(Eigen::VectorXf::Constant(weight.cols() * weight.rows(), 3));
  int k = 0;
  for (int j = 0; j < weight.rows(); j++ ) {
    for (int i = 0; i < weight.cols(); i++) {
      sparse.insert(k,k) = 2 * weight(i,j);
      if (j > 0 && i > 0) {
        sparse.insert(k, k - weight.cols() - 1) = -weight(i,j);
      }
      if (j < weight.rows()-1 && i < weight.cols() - 1) {
        sparse.insert(k, k + weight.cols() + 1) = -weight(i,j);
      }
      k++;
    }
  }
}
void vw::stereo::generate_laplacian4(ImageView<float> const& weight,
                                     Eigen::SparseMatrix<float> & sparse) {
  // These matrices are column major
  sparse.reserve(Eigen::VectorXf::Constant(weight.cols() * weight.rows(), 3));
  int k = 0;
  for (int j = 0; j < weight.rows(); j++ ) {
    for (int i = 0; i < weight.cols(); i++) {
      sparse.insert(k,k) = 2 * weight(i,j);
      if (j > 0 && i < weight.cols() - 1) {
        sparse.insert(k, k - weight.cols() + 1) = -weight(i,j);
      }
      if (j < weight.rows()-1 && i > 0) {
        sparse.insert(k, k + weight.cols() - 1) = -weight(i,j);
      }
      k++;
    }
  }
}

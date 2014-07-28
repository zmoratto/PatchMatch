#include <TVMin2.h>
#include <vw/Image/Filter.h>
#include <vw/Image/Statistics.h>

using namespace vw;

void backdiffx( ImageView<float> const& input,
                ImageView<float> & output ) {
  output.set_size(input.cols(), input.rows());
  crop(output, BBox2i(0,0,1,input.rows())) =
    crop(input, BBox2i(0,0,1,input.rows()));
  crop(output, BBox2i(1,0,input.cols()-1,input.rows())) =
    crop(input, BBox2i(0,0,input.cols()-1,input.rows()));
  output = input - output;
}

void backdiffy( ImageView<float> const& input,
                ImageView<float> & output ) {
  output.set_size(input.cols(), input.rows());
  crop(output, BBox2i(0,0,input.cols(),1)) =
    crop(input, BBox2i(0,0,input.cols(),1));
  crop(output, BBox2i(0,1,input.cols(),input.rows()-1)) =
    crop(input, BBox2i(0,0,input.cols(),input.rows()-1));
  output = input - output;
}

void forwarddiffx( ImageView<float> const& input,
                   ImageView<float> & output ) {
  output.set_size(input.cols(), input.rows());
  crop(output, BBox2i(0,0,input.cols()-1,input.rows())) =
    crop(input, BBox2i(1,0,input.cols()-1,input.rows()));
  crop(output, BBox2i(input.cols()-1,0,1,input.rows())) =
    crop(input, BBox2i(input.cols()-1,0,1,input.rows()));
  output -= input;
}

void forwarddiffy( ImageView<float> const& input,
                   ImageView<float> & output ) {
  output.set_size(input.cols(), input.rows());
  crop(output, BBox2i(0,0,input.cols(),input.rows()-1)) =
    crop(input, BBox2i(0,1,input.cols(),input.rows()-1));
  crop(output, BBox2i(0,input.rows()-1,input.cols(),1)) =
    crop(input, BBox2i(0,input.rows()-1,input.cols(),1));
  output -= input;
}

// To avoid casting higher for uint8 subtraction
template <class PixelT>
struct AbsDiffFunc : public vw::ReturnFixedType<PixelT> {
  inline PixelT operator()( PixelT const& a, PixelT const& b ) const {
    return fabs( a - b );
  }
};

void vw::stereo::imROF( ImageView<float> const& input,
                        float lambda, int iterations,
                        ImageView<float> & output ) {
  ImageView<float> p1( input.cols(), input.rows() );
  ImageView<float> p2( input.cols(), input.rows() );
  output.set_size( input.cols(), input.rows() );
  ImageView<float> old_output;
  ImageView<float> grad_u_x( input.cols(), input.rows() );
  ImageView<float> grad_u_y( input.cols(), input.rows() );
  ImageView<float> div_p( input.cols(), input.rows() );
  ImageView<float> grad_u_mag( input.cols(), input.rows() );
  double dt = lambda / 4;

  ImageView<float> tmp( input.cols(), input.rows() );

  for ( int i = 0; i < iterations; i++ ) {
    backdiffx(p1,div_p);
    backdiffy(p2,tmp);
    div_p += tmp;
    old_output = copy(output);
    output = input + div_p/lambda;

    /*
    // See if we should terminate
    float eps =
      sum_of_pixel_values(abs(output - old_output));
    std::cout << i << " " << eps << std::endl;
    */

    forwarddiffx(output, grad_u_x);
    forwarddiffy(output, grad_u_y);
    // Square is not defined
    for (int j = 0; j < p1.rows(); j++ ) {
      for ( int k = 0; k < p1.cols(); k++ ) {
        grad_u_mag(k,j) =
          sqrt(grad_u_x(k,j) * grad_u_x(k,j) +
               grad_u_y(k,j) * grad_u_y(k,j));
      }
    }
    tmp = float(1) + grad_u_mag * dt;
    p1 = dt * grad_u_x + p1;
    p2 = dt * grad_u_y + p2;
    // Element quotient is not defined for vectors as /
    for (int j = 0; j < p1.rows(); j++ ) {
      for ( int k = 0; k < p1.cols(); k++ ) {
        p1(k,j) = p1(k,j) / tmp(k,j);
      }
    }
    for (int j = 0; j < p1.rows(); j++ ) {
      for ( int k = 0; k < p1.cols(); k++ ) {
        p2(k,j) = p2(k,j) / tmp(k,j);
      }
    }
  }
}

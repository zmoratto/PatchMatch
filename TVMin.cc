#include <TVMin.h>

#include <vw/Image.h>
#include <vw/Math/Vector.h>

using namespace vw;

void backdiffx( ImageView<Vector2f> const& input,
                ImageView<Vector2f> & output ) {
  output.set_size(input.cols(), input.rows());
  crop(output, BBox2i(0,0,1,input.rows())) =
    crop(input, BBox2i(0,0,1,input.rows()));
  crop(output, BBox2i(1,0,input.cols()-1,input.rows())) =
    crop(input, BBox2i(0,0,input.cols()-1,input.rows()));
  output = input - output;
}

void backdiffy( ImageView<Vector2f> const& input,
                ImageView<Vector2f> & output ) {
  output.set_size(input.cols(), input.rows());
  crop(output, BBox2i(0,0,input.cols(),1)) =
    crop(input, BBox2i(0,0,input.cols(),1));
  crop(output, BBox2i(0,1,input.cols(),input.rows()-1)) =
    crop(input, BBox2i(0,0,input.cols(),input.rows()-1));
  output = input - output;
}

void forwarddiffx( ImageView<Vector2f> const& input,
                   ImageView<Vector2f> & output ) {
  output.set_size(input.cols(), input.rows());
  crop(output, BBox2i(0,0,input.cols()-1,input.rows())) =
    crop(input, BBox2i(1,0,input.cols()-1,input.rows()));
  crop(output, BBox2i(input.cols()-1,0,1,input.rows())) =
    crop(input, BBox2i(input.cols()-1,0,1,input.rows()));
  output -= input;
}

void forwarddiffy( ImageView<Vector2f> const& input,
                   ImageView<Vector2f> & output ) {
  output.set_size(input.cols(), input.rows());
  crop(output, BBox2i(0,0,input.cols(),input.rows()-1)) =
    crop(input, BBox2i(0,1,input.cols(),input.rows()-1));
  crop(output, BBox2i(0,input.rows()-1,input.cols(),1)) =
    crop(input, BBox2i(0,input.rows()-1,input.cols(),1));
  output -= input;
}

void imROF( ImageView<Vector2f> const& input,
            float lambda, int iterations,
            ImageView<Vector2f> & output ) {
  ImageView<Vector2f> p1( input.cols(), input.rows() );
  ImageView<Vector2f> p2( input.cols(), input.rows() );
  output.set_size( input.cols(), input.rows() );
  ImageView<Vector2f> grad_u_x( input.cols(), input.rows() );
  ImageView<Vector2f> grad_u_y( input.cols(), input.rows() );
  ImageView<Vector2f> div_p( input.cols(), input.rows() );
  ImageView<Vector2f> grad_u_mag( input.cols(), input.rows() );
  double dt = lambda / 4;

  ImageView<Vector2f> tmp( input.cols(), input.rows() );

  for ( int i = 0; i < iterations; i++ ) {
    backdiffx(p1,div_p);
    backdiffy(p2,tmp);
    div_p += tmp;
    output = input + div_p/lambda;
    forwarddiffx(output, grad_u_x);
    forwarddiffy(output, grad_u_y);
    // Square is not defined
    for (int j = 0; j < p1.rows(); j++ ) {
      for ( int k = 0; k < p1.cols(); k++ ) {
        grad_u_mag(k,j) =
          sqrt(elem_prod(grad_u_x(k,j),grad_u_x(k,j)) +
               elem_prod(grad_u_y(k,j),grad_u_y(k,j)));
      }
    }
    tmp = Vector2f(1, 1) + grad_u_mag * dt;
    p1 = dt * grad_u_x + p1;
    p2 = dt * grad_u_y + p2;
    // Element quotient is not defined for vectors as /
    for (int j = 0; j < p1.rows(); j++ ) {
      for ( int k = 0; k < p1.cols(); k++ ) {
        p1(k,j) = elem_quot(p1(k,j), tmp(k,j));
      }
    }
    for (int j = 0; j < p1.rows(); j++ ) {
      for ( int k = 0; k < p1.cols(); k++ ) {
        p2(k,j) = elem_quot(p2(k,j), tmp(k,j));
      }
    }
  }
}

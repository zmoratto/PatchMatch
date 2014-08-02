#include <TVMin3.h>
#include <vw/Image/Filter.h>
#include <vw/Image/Statistics.h>

#include <vw/FileIO.h>

using namespace vw;

void partial_x( ImageView<float> const& input,
                ImageView<float> & output ) {
  output.set_size(input.cols(), input.rows());
  fill(output,0);
  crop(output, BBox2i(0,0,input.cols()-1,input.rows()))
    += crop(input, BBox2i(1,0,input.cols()-1,input.rows()));
  crop(output, BBox2i(0,0,input.cols()-1,input.rows()))
    -= crop(input, BBox2i(0,0,input.cols()-1,input.rows()));
}

void partial_y( ImageView<float> const& input,
                ImageView<float> & output ) {
  output.set_size(input.cols(), input.rows());
  fill(output,0);
  crop(output, BBox2i(0,0,input.cols(),input.rows()-1))
    += crop(input, BBox2i(0,1,input.cols(),input.rows()-1));
  crop(output, BBox2i(0,0,input.cols(),input.rows()-1))
    -= crop(input, BBox2i(0,0,input.cols(),input.rows()-1));
}

void stereo::divergence( ImageView<float> const& input_x,
                         ImageView<float> const& input_y,
                         ImageView<float> & output ) {
  output.set_size(input_x.cols(), input_x.rows());
  fill(output,0);
  crop(output, BBox2i(1,0,input_x.cols()-1,input_x.rows())) -= crop(input_x, BBox2i(0,0,input_x.cols()-1,input_x.rows()));
  crop(output, BBox2i(0,0,input_x.cols()-1,input_x.rows())) += crop(input_x, BBox2i(0,0,input_x.cols()-1,input_x.rows()));
  crop(output, BBox2i(0,1,input_x.cols(),input_x.rows()-1)) -= crop(input_y, BBox2i(0,0,input_x.cols(),input_x.rows()-1));
  crop(output, BBox2i(0,0,input_x.cols(),input_x.rows()-1)) += crop(input_y, BBox2i(0,0,input_x.cols(),input_x.rows()-1));

  //output = -diff_x - diff_y;
  //output = input_x + input_y;
}

void stereo::gradient( ImageView<float> const& input,
                       ImageView<float> & output_x,
                       ImageView<float> & output_y) {
  partial_x(input, output_x);
  partial_y(input, output_y);
}

float calc_energy_ROF( ImageView<float> const& input,
                       ImageView<float> const& ref,
                       float lambda ) {
  ImageView<float> dx, dy;
  stereo::gradient(input, dx, dy);
  float e_reg = sum_of_pixel_values(sqrt(dx * dx + dy * dy));
  float e_data = 0.5 * lambda * sum_of_pixel_values((input - ref) * (input - ref));
  return e_reg + e_data;
}

void stereo::ROF( ImageView<float> const& input,
                  float lambda, int iterations,
                  float sigma, float tau, // These are gradient step sizes
                  ImageView<float> & output ) {
  // Allocate space for p, our hidden variable and u our output.
  ImageView<float> p_x(input.cols(), input.rows()),
    p_y(input.cols(), input.rows());
  output.set_size(input.cols(), input.rows());
  output = copy(input);
  ImageView<float> grad_u_x, grad_u_y;
  ImageView<float> div_p;
  stereo::gradient(output, p_x, p_y);
  for ( int i = 0; i < iterations; i++ ) {
    // Eqn 136
    stereo::gradient(output, grad_u_x, grad_u_y);
    p_x += sigma * grad_u_x;
    p_y += sigma * grad_u_y;

    // Eqn 137 (project to a unit sphere
    for (int j = 0; j < p_x.rows(); j++ ) {
      for (int i = 0; i < p_x.cols(); i++ ) {
        float mag =
          std::max(1.0, sqrt(p_x(i,j)*p_x(i,j) +
                             p_y(i,j)*p_y(i,j)));
        p_x(i,j) /= mag;
        p_y(i,j) /= mag;
      }
    }
    //std::cout << "g x: " << grad_u_x(10,10) << " y: " << grad_u_y(10,10) << " p x: " << p_x(10,10) << " y: " << p_y(10,10) << " " << output(10, 10) << std::endl;

    // Eqn 139
    stereo::divergence(p_x, p_y, div_p);
    output += tau * div_p + tau * lambda * input;
    // Eqn 139 (pt2)
    output /= (1 + tau * lambda);
    //std::cout << "Energy: " << calc_energy_ROF(output, input, lambda) << std::endl;
  }
}

void stereo::HuberROF( ImageView<float> const& input,
                       float lambda, int iterations,
                       float alpha, // Huber threshold coeff,
                       float sigma, float tau, // Gradient step sizes
                       ImageView<float> & output ) {
  // Allocate space for p, our hidden variable and u our output.
  ImageView<float> p_x(input.cols(), input.rows()),
    p_y(input.cols(), input.rows());
  output.set_size(input.cols(), input.rows());
  output = copy(input);
  ImageView<float> grad_u_x, grad_u_y;
  ImageView<float> div_p;
  gradient(output, p_x, p_y);
  for ( int i = 0; i < iterations; i++ ) {
    // Eqn 156 & 157
    gradient(output, grad_u_x, grad_u_y);
    p_x += sigma * grad_u_x;
    p_y += sigma * grad_u_y;
    p_x /= (1 + sigma * alpha);
    p_y /= (1 + sigma * alpha);

    // Eqn 158
    for (int j = 0; j < p_x.rows(); j++ ) {
      for (int i = 0; i < p_x.cols(); i++ ) {
        float mag =
          std::max(1.0, sqrt(p_x(i,j)*p_x(i,j) +
                             p_y(i,j)*p_y(i,j)));
        p_x(i,j) /= mag;
        p_y(i,j) /= mag;
      }
    }

    // Eqn 160
    divergence(p_x, p_y, div_p);
    output += tau * div_p + tau * lambda * input;
    output /= (1 + tau * lambda);
  }
}

void stereo::ROF_TVL1( ImageView<float> const& input,
                       float lambda, int iterations,
                       float sigma, float tau, // Gradient step sizes
                       ImageView<float> & output ) {
  // Allocate space for p, our hidden variable and u our output.
  ImageView<float> p_x(input.cols(), input.rows()),
    p_y(input.cols(), input.rows());
  output.set_size(input.cols(), input.rows());
  output = copy(input);
  ImageView<float> grad_u_x, grad_u_y;
  ImageView<float> div_p, q;
  stereo::gradient(output, p_x, p_y);
  q = copy(input);
  for ( int i = 0; i < iterations; i++ ) {
    // Eqn 185
    gradient(output, grad_u_x, grad_u_y);
    p_x += sigma * grad_u_x;
    p_y += sigma * grad_u_y;

    // Eqn 186
    for (int j = 0; j < p_x.rows(); j++ ) {
      for (int i = 0; i < p_x.cols(); i++ ) {
        float mag =
          std::max(1.0, sqrt(p_x(i,j)*p_x(i,j) +
                             p_y(i,j)*p_y(i,j)));
        p_x(i,j) /= mag;
        p_y(i,j) /= mag;
      }
    }

    // Eqn 188
    q += sigma * lambda * (output - input);

    // Eqn 189
    for (int j = 0; j < q.rows(); j++ ) {
      for (int i = 0; i < q.cols(); i++ ) {
        float mag =
          std::max(1.0, fabs(q(i,j)));
        q(i,j) /= mag;
      }
    }

    // Eqn 191
    stereo::divergence(p_x, p_y, div_p);
    output += tau * div_p - tau * lambda * q;
  }
}

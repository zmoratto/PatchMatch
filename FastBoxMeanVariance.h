#ifndef __VW_STEREO_FASTBOX_MEAN_VARIANCE_H__
#define __VW_STEREO_FASTBOX_MEAN_VARIANCE_H__

#include <vw/Math/Vector.h>
#include <vw/Image/ImageView.h>
#include <numeric>
#include <valarray>

namespace vw {
namespace stereo {


  template <class AccumulatorType, class ViewT>
  ImageView<Vector<AccumulatorType,2> >
  fast_box_mean_variance( ImageViewBase<ViewT> const& image, Vector2i const& kernel ) {

    // Sanity check, constants, and types
    VW_ASSERT( kernel[0] % 2 == 1 && kernel[1] % 2 == 1,
               ArgumentErr() << "fast_box_sum: Kernel input not sized with odd values." );

    typedef typename ViewT::pixel_accessor PAccT;
    typedef typename ViewT::pixel_type PixelT;
    typedef Vector<AccumulatorType,2> AccumT;

    ViewT input( image.impl() );
    VW_DEBUG_ASSERT( input.cols() >= kernel[0] && input.rows() >= kernel[1],
                     ArgumentErr() << "fast_box_sum: Image is not big enough for kernel." );

    ImageView<AccumT> output( input.cols() - kernel[0] + 1,
                              input.rows() - kernel[1] + 1 );
    AccumulatorType n = prod(kernel);
    typedef typename ImageView<AccumT>::pixel_accessor OAccT;

    // Start column sum
    //
    // ie sum the kernel height for the entire top lines of the image
    std::valarray<AccumulatorType> col_sum( input.cols() );
    std::valarray<AccumulatorType> col_sqsum( input.cols() );
    const AccumulatorType* col_sum_end = &col_sum[input.cols()];
    {
      PAccT row_input = input.origin();
      for ( int32 ky = kernel[1]; ky; --ky ) {
        PAccT col_input = row_input;
        AccumulatorType *sum = &col_sum[0];
        AccumulatorType *sqsum = &col_sqsum[0];
        while ( sum != col_sum_end ) {
          *sum++ += *col_input;
          *sqsum++ += AccumulatorType(*col_input)*AccumulatorType(*col_input);
          col_input.next_col();
        }
        row_input.next_row();
      }
    }

    // start rasterizing rows
    OAccT dst = output.origin();
    PAccT src_row_back = input.origin();
    PAccT src_row_front = input.origin().advance(0,kernel[1]);
    for ( int32 y = output.rows() - 1; y; --y ) {
      // Seed row sum
      AccumulatorType row_sum(0), row_sqsum(0);
      row_sum = std::accumulate(&col_sum[0],&col_sum[kernel[0]],
                                row_sum);
      row_sqsum = std::accumulate(&col_sqsum[0],&col_sqsum[kernel[0]],
                                  row_sqsum);

      // Sum down the row line
      AccumulatorType const *sumback = &col_sum[0], *sumfront = &col_sum[kernel[0]],
        *sqsumback = &col_sqsum[0], *sqsumfront = &col_sqsum[kernel[0]];
      while( sumfront != col_sum_end ) {
        *dst = AccumT(row_sum,row_sqsum)/n;
        (*dst)[1] = sqrt((*dst)[1] - (*dst)[0]*(*dst)[0]);
        dst.next_col();
        row_sum   += *sumfront++ - *sumback++;
        row_sqsum += *sqsumfront++ - *sqsumback++;
      }
      *dst = AccumT(row_sum,row_sqsum)/n;
      (*dst)[1] = sqrt((*dst)[1] - (*dst)[0]*(*dst)[0]);
      dst.next_col();

      // Update column sums
      PAccT src_col_back = src_row_back;
      PAccT src_col_front = src_row_front;
      AccumulatorType *sum   = &col_sum[0];
      AccumulatorType *sqsum = &col_sqsum[0];
      while ( sum != col_sum_end ) {
        *sum += *src_col_front; // We do this in 2 lines to avoid casting.
        *sum -= *src_col_back;  // I'm unsure if the assembly is still doing that.
        *sqsum += AccumulatorType(*src_col_front)*AccumulatorType(*src_col_front);
        *sqsum -= AccumulatorType(*src_col_back)*AccumulatorType(*src_col_back);

        sum++;
        sqsum++;
        src_col_back.next_col();
        src_col_front.next_col();
      }

      // Update row iterators
      src_row_back.next_row();
      src_row_front.next_row();
    }

    { // Perform last sum down the line
      // Seed row sum
      AccumulatorType row_sum(0), row_sqsum(0);
      row_sum =
        std::accumulate(&col_sum[0],&col_sum[kernel[0]],
                        row_sum);
      row_sqsum =
        std::accumulate(&col_sqsum[0],&col_sqsum[kernel[0]],
                        row_sqsum);

      // Sum down the row line
      AccumulatorType const *sumback = &col_sum[0], *sumfront = &col_sum[kernel[0]];
      AccumulatorType const *sqsumback = &col_sum[0], *sqsumfront = &col_sum[kernel[0]];
      while( sumfront != col_sum_end ) {
        *dst = AccumT(row_sum,row_sqsum)/n;
        (*dst)[1] = sqrt((*dst)[1] - (*dst)[0]*(*dst)[0]);
        dst.next_col();
        row_sum += *sumfront++ - *sumback++;
        row_sqsum += *sqsumfront++ - *sqsumback++;
      }
      *dst = AccumT(row_sum,row_sqsum)/n;
      (*dst)[1] = sqrt((*dst)[1] - (*dst)[0]*(*dst)[0]);
    }

    return output;
  }

}} // vw::stereo

#endif//__VW_STEREO_FASTBOX_MEAN_VARIANCE_H__

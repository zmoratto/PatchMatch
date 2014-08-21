// __BEGIN_LICENSE__
//  Copyright (c) 2006-2013, United States Government as represented by the
//  Administrator of the National Aeronautics and Space Administration. All
//  rights reserved.
//
//  The NASA Vision Workbench is licensed under the Apache License,
//  Version 2.0 (the "License"); you may not use this file except in
//  compliance with the License. You may obtain a copy of the License at
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// __END_LICENSE__


#ifndef __VW_STEREO_MAPPING_PYRAMID_CORRELATION_VIEW_H__
#define __VW_STEREO_MAPPING_PYRAMID_CORRELATION_VIEW_H__

#include <vw/Core/Exception.h>
#include <vw/Core/Stopwatch.h>
#include <vw/Core/Thread.h>
#include <vw/Core/Debugging.h>
#include <vw/Math/BBox.h>
#include <vw/Image/Algorithms.h>
#include <vw/Image/AlgorithmFunctions.h>
#include <vw/Image/PerPixelAccessorViews.h>
#include <vw/FileIO.h>
#include <vw/Stereo/Correlation.h>
#include <vw/Stereo/Correlate.h>
#include <vw/Stereo/DisparityMap.h>
#include <vw/Stereo/PreFilter.h>
#include <boost/foreach.hpp>

//#define WRITE_DEBUG 1

namespace vw {
  namespace stereo {

    /// An image view for performing pyramid image correlation (Faster
    /// than CorrelationView).
    template <class Image1T, class Image2T, class Mask1T, class Mask2T, class PreFilterT>
    class MappingPyramidCorrelationView : public ImageViewBase<MappingPyramidCorrelationView<Image1T,Image2T, Mask1T, Mask2T, PreFilterT> > {

      Image1T m_left_image;
      Image2T m_right_image;
      Mask1T  m_left_mask;
      Mask2T  m_right_mask;
      PreFilterT m_prefilter;
      BBox2i m_search_region;
      Vector2i m_kernel_size;
      CostFunctionType m_cost_type;
      int m_corr_timeout;
      // How long it takes to do one corr op with given kernel and cost function
      float m_consistency_threshold; // < 0 = means don't do a consistency check
      int32 m_max_level_by_search;

      struct SubsampleMaskByTwoFunc : public ReturnFixedType<uint8> {
        BBox2i work_area() const { return BBox2i(0,0,2,2); }

        template <class PixelAccessorT>
        typename boost::remove_reference<typename PixelAccessorT::pixel_type>::type
        operator()( PixelAccessorT acc ) const {

          typedef typename PixelAccessorT::pixel_type PixelT;

          uint8 count = 0;
          if ( *acc ) count++;
          acc.next_col();
          if ( *acc ) count++;
          acc.advance(-1,1);
          if ( *acc ) count++;
          acc.next_col();
          if ( *acc ) count++;
          if ( count > 1 )
            return PixelT(ScalarTypeLimits<PixelT>::highest());
          return PixelT();
        }
      };

      template <class ViewT>
      SubsampleView<UnaryPerPixelAccessorView<EdgeExtensionView<ViewT,ZeroEdgeExtension>, SubsampleMaskByTwoFunc> >
      subsample_mask_by_two( ImageViewBase<ViewT> const& input ) const {
        return subsample(per_pixel_accessor_filter(input.impl(), SubsampleMaskByTwoFunc()),2);
      }

      template <class ImageT, class TransformT>
      TransformView<InterpolationView<ImageT, BilinearInterpolation>, TransformT>
      inline transform_no_edge( ImageViewBase<ImageT> const& v,
                                TransformT const& transform_func ) const {
        return TransformView<InterpolationView<ImageT, BilinearInterpolation>, TransformT>(InterpolationView<ImageT, BilinearInterpolation>(v.impl()), transform_func);
      }

      void blur_disparity(ImageView<PixelMask<Vector2f> >& sf_disparity,
                          BBox2i const& disparity_bounds) const {
        select_channel(sf_disparity,0) =
          clamp(gaussian_filter(select_channel(sf_disparity,0),5),
                disparity_bounds.min()[0],
                disparity_bounds.max()[0]);
        select_channel(sf_disparity,1) =
          clamp(gaussian_filter(select_channel(sf_disparity,1),5),
                disparity_bounds.min()[1],
                disparity_bounds.max()[1]);
      }

      void copy_valid(ImageView<PixelMask<Vector2f> >& destination,
                      ImageView<PixelMask<Vector2f> >& source) const {
        for (int j = 0; j < destination.rows(); j++ ) {
          for (int i = 0; i < destination.cols(); i++ ) {
            if (is_valid(source(i,j))) {
              destination(i,j) = source(i,j);
            }
          }
        }
      }

    public:
      typedef PixelMask<Vector2f> pixel_type;
      typedef PixelMask<Vector2f> result_type;
      typedef ProceduralPixelAccessor<MappingPyramidCorrelationView> pixel_accessor;

      MappingPyramidCorrelationView( ImageViewBase<Image1T> const& left,
                                     ImageViewBase<Image2T> const& right,
                                     ImageViewBase<Mask1T> const& left_mask,
                                     ImageViewBase<Mask2T> const& right_mask,
                                     PreFilterBase<PreFilterT> const& prefilter,
                                     BBox2i const& search_region, Vector2i const& kernel_size,
                                     CostFunctionType cost_type,
                                     float consistency_threshold,
                                     int32 max_pyramid_levels) :
      m_left_image(left.impl()), m_right_image(right.impl()),
        m_left_mask(left_mask.impl()), m_right_mask(right_mask.impl()),
        m_prefilter(prefilter.impl()), m_search_region(search_region), m_kernel_size(kernel_size),
        m_cost_type(cost_type),
        m_consistency_threshold(consistency_threshold){
        // Calculating max pyramid levels according to the supplied
        // search region.
        int32 largest_search = max( search_region.size() );
        m_max_level_by_search = std::floor(std::log(float(largest_search))/std::log(2.0f)) - 1;
        if ( m_max_level_by_search > max_pyramid_levels )
          m_max_level_by_search = max_pyramid_levels;
        if ( m_max_level_by_search < 0 )
          m_max_level_by_search = 0;
      }

      // Standard required ImageView interfaces
      inline int32 cols() const { return m_left_image.cols(); }
      inline int32 rows() const { return m_left_image.rows(); }
      inline int32 planes() const { return 1; }

      inline pixel_accessor origin() const { return pixel_accessor( *this, 0, 0 ); }
      inline pixel_type operator()( int32 /*i*/, int32 /*j*/, int32 /*p*/ = 0) const {
        vw_throw( NoImplErr() << "MappingPyramidCorrelationView::operator()(....) has not been implemented." );
        return pixel_type();
      }

      // Block rasterization section that does actual work
      typedef CropView<ImageView<pixel_type> > prerasterize_type;
      inline prerasterize_type prerasterize(BBox2i const& bbox) const {

        std::cout << "\tRequesting bbox: " << bbox << std::endl;

#if VW_DEBUG_LEVEL > 0
        Stopwatch watch;
        watch.start();
#endif

#ifdef WRITE_DEBUG
        std::ostringstream tags;
        tags << bbox.min().x() << "_" << bbox.min().y() << "_";
        std::string tag = tags.str();
#endif

        // 1.0) Determining the number of levels to process
        //      There's a maximum base on kernel size. There's also
        //      maximum defined by the search range. Here we determine
        //      the maximum based on kernel size and current bbox.
        int32 smallest_bbox = math::min(bbox.size());
        int32 largest_kernel = math::max(m_kernel_size);
        int32 max_pyramid_levels = std::floor(log(smallest_bbox)/log(2.0f) - log(largest_kernel)/log(2.0f));
        int32 max_level_by_size = std::ceil(log(smallest_bbox / 64.0) / log(2.0f));
        std::cout << "\t" << max_pyramid_levels << " " << max_level_by_size << std::endl;
        max_pyramid_levels = std::min(max_pyramid_levels, m_max_level_by_search);
        max_pyramid_levels = std::min(max_pyramid_levels, max_level_by_size);
        if ( max_pyramid_levels < 1 )
          max_pyramid_levels = 0;
        Vector2i half_kernel = m_kernel_size/2;

        // 2.0) Build the pyramid
        std::vector<ImageView<typename Image1T::pixel_type> > left_pyramid(max_pyramid_levels + 1 );
        std::vector<ImageView<typename Image2T::pixel_type> > right_pyramid(max_pyramid_levels + 1 );
        std::vector<ImageView<typename Mask1T::pixel_type> > left_mask_pyramid(max_pyramid_levels + 1 );
        std::vector<ImageView<typename Mask2T::pixel_type> > right_mask_pyramid(max_pyramid_levels + 1 );
        std::vector<BBox2i> left_roi(max_pyramid_levels + 1);
        std::vector<BBox2i> right_roi(max_pyramid_levels + 1);

        int32 max_upscaling = 1 << max_pyramid_levels;
        {
          left_roi[0] = bbox;
          left_roi[0].min() -= half_kernel * max_upscaling;
          left_roi[0].max() += half_kernel * max_upscaling;
          right_roi[0] = left_roi[0] + m_search_region.min();
          right_roi[0].max() += m_search_region.size() + Vector2i(max_upscaling,max_upscaling);
          left_pyramid[0] = crop(edge_extend(m_left_image),left_roi[0]);
          right_pyramid[0] = crop(edge_extend(m_right_image),right_roi[0]);
          left_mask_pyramid[0] =
            crop(edge_extend(m_left_mask,ConstantEdgeExtension()),
                 left_roi[0]);
          right_mask_pyramid[0] =
            crop(edge_extend(m_right_mask,ConstantEdgeExtension()),
                 right_roi[0]);

#if VW_DEBUG_LEVEL > 0
          VW_OUT(DebugMessage,"stereo") << " > Left ROI: " << left_roi[0]
                                        << "\n > Right ROI: " << right_roi[0] << "\n";
#endif

          // Fill in the nodata of the left and right images with a mean
          // pixel value. This helps with the edge quality of a DEM.
          typename Image1T::pixel_type left_mean;
          typename Image2T::pixel_type right_mean;
          try {
            left_mean =
              mean_pixel_value(subsample(copy_mask(left_pyramid[0],
                                                   create_mask(left_mask_pyramid[0],0)),2));
            right_mean =
              mean_pixel_value(subsample(copy_mask(right_pyramid[0],
                                                   create_mask(right_mask_pyramid[0],0)),2));
          } catch ( const ArgumentErr& err ) {
            // Mean pixel value will throw an argument error if there
            // are no valid pixels. If that happens, it means either the
            // left or the right image is full masked.
#if VW_DEBUG_LEVEL > 0
            watch.stop();
            double elapsed = watch.elapsed_seconds();
            vw_out(DebugMessage,"stereo")
              << "Tile " << bbox << " has no data. Processed in "
              << elapsed << " s\n";
#endif
            return prerasterize_type(ImageView<pixel_type>(bbox.width(),
                                                           bbox.height()),
                                     -bbox.min().x(), -bbox.min().y(),
                                     cols(), rows() );
          }
          left_pyramid[0] = apply_mask(copy_mask(left_pyramid[0],create_mask(left_mask_pyramid[0],0)), left_mean );
          right_pyramid[0] = apply_mask(copy_mask(right_pyramid[0],create_mask(right_mask_pyramid[0],0)), right_mean );

          // Don't actually need the whole over cropped disparity
          // mask. We only need the active region. I over cropped before
          // just to calculate the mean color value options.
          BBox2i right_mask = bbox + m_search_region.min();
          right_mask.max() += m_search_region.size();
          left_mask_pyramid[0] =
            crop(left_mask_pyramid[0], bbox - left_roi[0].min());
          right_mask_pyramid[0] =
            crop(right_mask_pyramid[0], right_mask - right_roi[0].min());

          // Szeliski's book recommended this simple kernel. This
          // operation is quickly becoming a time sink, we might
          // possibly want to write an integer optimized version.
          std::vector<typename DefaultKernelT<typename Image1T::pixel_type>::type > kernel(5);
          kernel[0] = kernel[4] = 1.0/16.0;
          kernel[1] = kernel[3] = 4.0/16.0;
          kernel[2] = 6.0/16.0;

          // Build the pyramid first and then apply the filter to each
          // level.

          // Move to the coordinate frame defined by a purely positive
          // search range.
          right_roi[0] -= m_search_region.min();
          // Move the coordinate frame to be relative to the query point
          left_roi[0] -= bbox.min();
          right_roi[0] -= bbox.min();

          std::cout << "\tL0 " << left_roi[0] << " " << right_roi[0] << std::endl;
          for ( int32 i = 0; i < max_pyramid_levels; ++i ) {
            left_pyramid[i+1] = subsample(separable_convolution_filter(left_pyramid[i],kernel,kernel),2);
            right_pyramid[i+1] = subsample(separable_convolution_filter(right_pyramid[i],kernel,kernel),2);

            // This fancy arithmetic is just a version of BBox2i() / 2
            // that produces results that match subsample()'s actual
            // output image sizes.
            left_roi[i+1] = BBox2i(left_roi[i].min().x() / 2, left_roi[i].min().y() / 2,
                                   1 + (left_roi[i].width() - 1) / 2,
                                   1 + (left_roi[i].height() - 1) / 2);
            right_roi[i+1] = BBox2i(right_roi[i].min().x() / 2, right_roi[i].min().y() / 2,
                                    1 + (right_roi[i].width() - 1) / 2,
                                    1 + (right_roi[i].height() - 1) / 2);
            std::cout << "\tL" << i + 1 << " " << left_roi[i+1] << " " << right_roi[i+1] << std::endl;
            VW_ASSERT(left_roi[i+1].size() == Vector2i(left_pyramid[i+1].cols(),
                                                       left_pyramid[i+1].rows()),
                      MathErr() << "Left ROI doesn't match pyramid image size");
            VW_ASSERT(right_roi[i+1].size() == Vector2i(right_pyramid[i+1].cols(),
                                                        right_pyramid[i+1].rows()),
                      MathErr() << "Right ROI doesn't match pyramid image size" << right_roi[i+1] << " " << bounding_box(right_pyramid[i+1]));

            left_pyramid[i] = m_prefilter.filter(left_pyramid[i]);
            right_pyramid[i] = m_prefilter.filter(right_pyramid[i]);

#ifdef WRITE_DEBUG
            std::ostringstream ostr;
            ostr << i;
            write_image(tag+"pyramid_"+ostr.str()+"-L.tif", left_pyramid[i]);
#endif

            left_mask_pyramid[i+1] = subsample_mask_by_two(left_mask_pyramid[i]);
            right_mask_pyramid[i+1] = subsample_mask_by_two(right_mask_pyramid[i]);
          }
          left_pyramid[max_pyramid_levels] = m_prefilter.filter(left_pyramid[max_pyramid_levels]);
          right_pyramid[max_pyramid_levels] = m_prefilter.filter(right_pyramid[max_pyramid_levels]);
        }

        // 3.0) Actually perform correlation now
        // 3.1) Perform a dense correlation at the top most image using the original unwarped images
        int32 scaling = 1 << max_pyramid_levels;
        Vector2i region_offset = max_upscaling * half_kernel / scaling;

        BBox2i left_region = bounding_box(left_mask_pyramid[max_pyramid_levels]) + region_offset;
        left_region.min() -= half_kernel;
        left_region.max() += half_kernel;
        BBox2i right_region = left_region;
        right_region.max() += m_search_region.size() / max_upscaling + Vector2i(1,1);

        std::cout << "\tLeft region: " << left_region << " " << bounding_box(left_pyramid[max_pyramid_levels]) << std::endl;
        std::cout << "\tRight region: " << right_region << " " << bounding_box(right_pyramid[max_pyramid_levels]) << std::endl;


        Vector2i top_level_search = m_search_region.size() / max_upscaling + Vector2i(1,1);

        ImageView<PixelMask<Vector2i> > disparity =
          calc_disparity(m_cost_type,
                         left_pyramid[max_pyramid_levels],
                         right_pyramid[max_pyramid_levels],
                         /* This ROI is actually the active area we'll
                            work over the left image image. That is
                            including the kernel space. */
                         left_roi[max_pyramid_levels]
                         - left_roi[max_pyramid_levels].min(),
                         top_level_search, m_kernel_size);
        ImageView<PixelMask<Vector2i> > rl_disparity =
          calc_disparity(m_cost_type,
                         right_pyramid[max_pyramid_levels],
                         crop(edge_extend(left_pyramid[max_pyramid_levels]),
                              left_roi[max_pyramid_levels] - left_roi[max_pyramid_levels].min()
                              - top_level_search),
                         right_roi[max_pyramid_levels] - right_roi[max_pyramid_levels].min(),
                         top_level_search, m_kernel_size)
          - pixel_type(top_level_search);
        stereo::cross_corr_consistency_check(disparity,
                                             rl_disparity,
                                             m_consistency_threshold, false);
#ifdef WRITE_DEBUG
        write_image(tag+"initial-L.tif", crop(left_pyramid[max_pyramid_levels], left_region));
        write_image(tag+"initial-R.tif", crop(right_pyramid[max_pyramid_levels], right_region));
        write_image(tag+"initial-D.tif", disparity);
#endif

        const BBox2i additive_search_range(-8, -8, 16, 16);
        const Vector2i surface_fit_tile(32, 32);

        ImageView<PixelMask<Vector2f> > smooth_disparity =
          block_rasterize(stereo::surface_fit(disparity),
                          surface_fit_tile, 2);
        blur_disparity(smooth_disparity,
                       BBox2i(Vector2i(0, 0),
                              m_search_region.size() / max_upscaling));
#ifdef WRITE_DEBUG
        write_image(tag+"initial_smooth-D.tif", smooth_disparity);
#endif


        // 3.2) Starting working through the lower levels where we
        // first map the right image to the left image, the correlate.
        ImageView<PixelMask<Vector2f> > super_disparity;
        for ( int32 level = max_pyramid_levels - 1; level >= 0; --level) {
          scaling = 1 << level;
          std::ostringstream ostr;
          ostr << "level" << level << "_";

          // The active area is less than what we have actually
          // rendered in the pyramid tree. The reason is that the
          // pyramid is padded by a kernel width at the top most
          // level. At this point though, we only need a kernel
          // padding at the scale we are currently at.
          BBox2i active_left_roi = bounding_box(left_mask_pyramid[level]);
          active_left_roi.min() -= half_kernel;
          active_left_roi.max() += half_kernel;

          BBox2i active_right_roi = active_left_roi;
          active_right_roi.max() += additive_search_range.size();
          std::cout << "\tL" << level << " Active ROI: " << active_left_roi << " " << active_right_roi << std::endl;

          // Upsample the previous disparity and then extrapolate the
          // disparity out so we can fill in the whole right roi that
          // we need.
          super_disparity =
            2 * crop(resample(smooth_disparity, 2, 2), bounding_box(left_mask_pyramid[level]))
            + PixelMask<Vector2f>(additive_search_range.min());
#ifdef WRITE_DEBUG
          write_image(tag+ostr.str()+"super-D.tif", super_disparity);
#endif
          ImageView<PixelMask<Vector2f> > super_disparity_exp =
            crop(edge_extend(super_disparity), active_right_roi);
#ifdef WRITE_DEBUG
          write_image(tag+ostr.str()+"super_exp-D.tif", super_disparity_exp);

          write_image(tag+ostr.str()+"L.tif",
                      crop(left_pyramid[level], active_left_roi - left_roi[level].min()));
          write_image(tag+ostr.str()+"R.tif",
                      crop(right_pyramid[level], active_right_roi - right_roi[level].min()));
#endif
          ImageView<float> right_t;
          {
            vw::Timer timer("Transform right");
            right_t =
              crop(transform_no_edge(crop(edge_extend(right_pyramid[level]),
                                          active_right_roi.min().x() - right_roi[level].min().x(),
                                          active_right_roi.min().y() - right_roi[level].min().y(),
                                          1, 1),
                                     stereo::DisparityTransform(super_disparity_exp)),
                   active_right_roi - active_right_roi.min());
#ifdef WRITE_DEBUG
            write_image(tag+ostr.str()+"R_tfm.tif", right_t);
#endif
          }

          {
            vw::Timer timer("Disparity LR");
            disparity =
              calc_disparity(m_cost_type,
                             crop(left_pyramid[level], active_left_roi - left_roi[level].min()),
                             right_t, active_left_roi - active_left_roi.min(),
                             additive_search_range.size(), m_kernel_size);
          }
          {
            vw::Timer timer("Disparity RL");
            rl_disparity =
              calc_disparity(m_cost_type,
                             right_t,
                             crop(edge_extend(left_pyramid[level]),
                                  active_left_roi - left_roi[level].min()
                                  - additive_search_range.size()),
                             bounding_box(right_t),
                             additive_search_range.size(), m_kernel_size)
              - pixel_type(additive_search_range.size());
          }
#ifdef WRITE_DEBUG
          write_image(tag+ostr.str()+"D.tif", disparity);
          write_image(tag+ostr.str()+"invD.tif", disparity);
#endif
          stereo::cross_corr_consistency_check(disparity, rl_disparity,
                                               m_consistency_threshold, false);
#ifdef WRITE_DEBUG
          write_image(tag+ostr.str()+"maskD.tif", disparity);
#endif
          super_disparity += pixel_cast<PixelMask<Vector2f> >(disparity);
#ifdef WRITE_DEBUG
          write_image(tag+ostr.str()+"mask_addD.tif", super_disparity);
#endif
          if (level != 0) {
            smooth_disparity =
              block_rasterize(stereo::surface_fit(super_disparity),
                              surface_fit_tile, 2);
#ifdef WRITE_DEBUG
            write_image(tag+ostr.str()+"fit-D.tif", smooth_disparity);
#endif
            copy_valid(smooth_disparity, super_disparity);
#ifdef WRITE_DEBUG
            write_image(tag+ostr.str()+"fit_plus-D.tif", smooth_disparity);
#endif
            blur_disparity(smooth_disparity,
                           BBox2i(Vector2i(),
                                  m_search_region.size() / scaling));
#ifdef WRITE_DEBUG
            write_image(tag+ostr.str()+"smooth-D.tif", smooth_disparity);
#endif
          }
        }

#if VW_DEBUG_LEVEL > 0
        watch.stop();
        double elapsed = watch.elapsed_seconds();
        vw_out(DebugMessage,"stereo") << "Tile " << bbox << " processed in "
                                      << elapsed << " s\n";
#endif

        // 5.0) Reposition our result back into the global
        // solution. Also we need to correct for the offset we applied
        // to the search region.
        return prerasterize_type(super_disparity + pixel_type(m_search_region.min()),
                                 -bbox.min().x(), -bbox.min().y(),
                                 cols(), rows() );
      }

      template <class DestT>
      inline void rasterize(DestT const& dest, BBox2i const& bbox) const {
        vw::rasterize(prerasterize(bbox), dest, bbox);
      }
    };

    template <class Image1T, class Image2T, class Mask1T, class Mask2T, class PreFilterT>
    MappingPyramidCorrelationView<Image1T,Image2T,Mask1T,Mask2T,PreFilterT>
    mapping_pyramid_correlate( ImageViewBase<Image1T> const& left,
                               ImageViewBase<Image2T> const& right,
                               ImageViewBase<Mask1T> const& left_mask,
                               ImageViewBase<Mask2T> const& right_mask,
                               PreFilterBase<PreFilterT> const& filter,
                               BBox2i const& search_region, Vector2i const& kernel_size,
                               CostFunctionType cost_type,
                               float consistency_threshold,
                               int32 max_pyramid_levels) {
      typedef MappingPyramidCorrelationView<Image1T,Image2T,Mask1T,Mask2T,PreFilterT> result_type;
      return result_type( left.impl(), right.impl(), left_mask.impl(),
                          right_mask.impl(), filter.impl(), search_region,
                          kernel_size, cost_type,
                          consistency_threshold, max_pyramid_levels );
    }

  }} // namespace vw::stereo

#endif//__VW_STEREO_MAPPING_PYRAMID_CORRELATION_VIEW_H__

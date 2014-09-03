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
      int32 m_padding;

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
                                     int32 padding) :
      m_left_image(left.impl()), m_right_image(right.impl()),
        m_left_mask(left_mask.impl()), m_right_mask(right_mask.impl()),
        m_prefilter(prefilter.impl()), m_search_region(search_region), m_kernel_size(kernel_size),
        m_cost_type(cost_type),
        m_consistency_threshold(consistency_threshold),
        m_padding(padding) {
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

        BBox2i bbox_exp = bbox;
        bbox_exp.expand(m_padding);

#if VW_DEBUG_LEVEL > 0
        Stopwatch watch;
        watch.start();
#endif

        // 1.0) Determining the number of levels to process
        //      There's a maximum base on kernel size. There's also
        //      maximum defined by the search range. Here we determine
        //      the maximum based on kernel size and current bbox.
        int32 smallest_bbox = math::min(bbox_exp.size());
        int32 largest_bbox = math::max(bbox_exp.size());
        int32 largest_kernel = math::max(m_kernel_size);
        int32 max_pyramid_levels = std::floor(log(smallest_bbox)/log(2.0f) - log(largest_kernel)/log(2.0f));
        int32 max_level_by_size = std::ceil(log(largest_bbox / 64.0) / log(2.0f));
        max_pyramid_levels = std::min(max_pyramid_levels, max_level_by_size);
        if ( max_pyramid_levels < 1 )
          max_pyramid_levels = 1;
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
          left_roi[0] = bbox_exp;
          left_roi[0].min() -= half_kernel * max_upscaling;
          left_roi[0].max() += half_kernel * max_upscaling;
          right_roi[0] = left_roi[0] + m_search_region.min();
          right_roi[0].max() += m_search_region.size() + Vector2i(max_upscaling,max_upscaling);
          left_pyramid[0] = crop(edge_extend(m_left_image),left_roi[0]);
          right_pyramid[0] = crop(edge_extend(m_right_image),right_roi[0]);
          left_mask_pyramid[0] =
            crop(edge_extend(m_left_mask, ConstantEdgeExtension()), left_roi[0]);
          right_mask_pyramid[0] =
            crop(edge_extend(m_right_mask, ConstantEdgeExtension()), right_roi[0]);

          std::cout << "LROI in Global: " << left_roi[0] << std::endl;
          std::cout << "RROI in Global: " << right_roi[0] << std::endl;

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
            // left or the right image is fullly masked.
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
          BBox2i right_mask = bbox_exp + m_search_region.min();
          right_mask.max() += m_search_region.size();
          left_mask_pyramid[0] =
            crop(left_mask_pyramid[0], bbox_exp - left_roi[0].min());
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
          left_roi[0] -= bbox_exp.min();
          right_roi[0] -= bbox_exp.min();

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
            VW_ASSERT(left_roi[i+1].size() == Vector2i(left_pyramid[i+1].cols(),
                                                       left_pyramid[i+1].rows()),
                      MathErr() << "Left ROI doesn't match pyramid image size");
            VW_ASSERT(right_roi[i+1].size() == Vector2i(right_pyramid[i+1].cols(),
                                                        right_pyramid[i+1].rows()),
                      MathErr() << "Right ROI doesn't match pyramid image size" << right_roi[i+1] << " " << bounding_box(right_pyramid[i+1]));

            left_pyramid[i] = m_prefilter.filter(left_pyramid[i]);
            right_pyramid[i] = m_prefilter.filter(right_pyramid[i]);

            left_mask_pyramid[i+1] = subsample_mask_by_two(left_mask_pyramid[i]);
            right_mask_pyramid[i+1] = subsample_mask_by_two(right_mask_pyramid[i]);
          }
          left_pyramid[max_pyramid_levels] = m_prefilter.filter(left_pyramid[max_pyramid_levels]);
          right_pyramid[max_pyramid_levels] = m_prefilter.filter(right_pyramid[max_pyramid_levels]);
        }

        std::cout << "BBox:     " << bbox << std::endl;
        std::cout << "BBox Exp: " << bbox_exp << std::endl;
        std::cout << "Padding:  " << m_padding << std::endl;
        std::cout << "KernlSz:  " << m_kernel_size << std::endl;
        std::cout << "SearchR:  " << m_search_region << std::endl;
        std::cout << "PyramidL: " << max_pyramid_levels << std::endl;
        std::cout << "LeftRoi0: " << left_roi[0] << std::endl;
        std::cout << "RighRoi0: " << right_roi[0] << std::endl;
        std::cout << "LeftRoi1: " << left_roi[1] << std::endl;
        std::cout << "RighRoi1: " << right_roi[1] << std::endl;

        // 3.0) Actually perform correlation now
        Vector2i top_level_search = m_search_region.size() / max_upscaling + Vector2i(1,1);

        // 3.1) Perform a dense correlation at the top most image
        // using the original unwarped images. This is the only time
        // we'll actually use the full search range.
        ImageView<PixelMask<Vector2i> > disparity, rl_disparity;
        {
          disparity =
            calc_disparity(m_cost_type,
                           left_pyramid[max_pyramid_levels],
                           right_pyramid[max_pyramid_levels],
                           /* This ROI is actually the active area we'll
                              work over the left image image. That is
                              including the kernel space. */
                           left_roi[max_pyramid_levels]
                           - left_roi[max_pyramid_levels].min(),
                           top_level_search, m_kernel_size);
          rl_disparity =
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
          write_image("leveltop-delta-D.tif", disparity);
        }

        // This numbers we're picked heuristically. If the additive
        // search range was smaller though .. we would process a lot
        // faster.
        const BBox2i additive_search_range(-8, -8, 16, 16);
        const Vector2i surface_fit_tile(32, 32);

        // Solve for an 'idealized' and smooth version of the
        // disparity so we have something nice to correlate against.
        ImageView<PixelMask<Vector2f> > smooth_disparity =
          block_rasterize(stereo::surface_fit(disparity),
                          surface_fit_tile, 2);
        blur_disparity(smooth_disparity,
                       BBox2i(Vector2i(0, 0),
                              m_search_region.size() / max_upscaling));

        // 3.2) Starting working through the lower levels where we
        // first map the right image to the left image, the correlate.
        ImageView<PixelMask<Vector2f> > super_disparity, super_disparity_exp;
        ImageView<float> right_t;
        for ( int32 level = max_pyramid_levels - 1; level > 0; --level) {
          int32 scaling = 1 << level;
          Vector2i output_size = Vector2i(1,1) + (bbox_exp.size() - Vector2i(1,1)) / scaling;
          std::cout << "Level " << level << " ---------------------------" << std::endl;
          std::cout << output_size << std::endl;

          std::ostringstream ostr;
          ostr << "level" << level;

          // The active area is less than what we have actually
          // rendered in the pyramid tree. The reason is that the
          // pyramid is padded by a kernel width at the top most
          // level. At this point though, we only need a kernel
          // padding at the scale we are currently at.
          BBox2i active_left_roi(Vector2i(), output_size);
          active_left_roi.min() -= half_kernel;
          active_left_roi.max() += half_kernel;
          std::cout << "Active L ROI: " << active_left_roi << std::endl;

          BBox2i active_right_roi = active_left_roi;
          active_right_roi.max() += additive_search_range.max();
          active_right_roi.min() += additive_search_range.min();
          std::cout << "Active R ROI: " << active_right_roi << std::endl;
          std::cout << "L_ROI: " << left_roi[level] << std::endl;
          std::cout << "R_ROI: " << right_roi[level] << std::endl;
          std::cout << "Input Disp Size: " << bounding_box(smooth_disparity) << std::endl;

          // Upsample the previous disparity and then extrapolate the
          // disparity out so we can fill in the whole right roi that
          // we need.
          super_disparity_exp =
            crop(edge_extend(2 * crop(resample(smooth_disparity, 2, 2),
                                      BBox2i(Vector2i(), output_size))),
                 active_right_roi);
          std::cout << "Upsampled Disp Size: " << bounding_box(super_disparity_exp) << std::endl;

          // The center crop (right after the edge extend) is because
          // we actually want to resample just a half kernel out from
          // the active area we care about. However the pyramid
          // actually has a lot more imagery than we need as it was
          // padded for a half kernel at the highest level.
          //
          // The crop size doesn't matter for the inner round since we
          // need it only to shift the origin and because we are using
          // a version of transform that doesn't call edge extend.
          right_t =
              crop(transform_no_edge
                   (crop(edge_extend(right_pyramid[level]),
                         active_right_roi.min().x() - right_roi[level].min().x(),
                         active_right_roi.min().y() - right_roi[level].min().y(), 1, 1),
                    stereo::DisparityTransform(super_disparity_exp)),
                   active_right_roi - active_right_roi.min());

          disparity =
            calc_disparity(m_cost_type,
                           crop(left_pyramid[level], active_left_roi - left_roi[level].min()),
                           right_t, active_left_roi - active_left_roi.min(),
                           additive_search_range.size(), m_kernel_size);

          write_image(ostr.str()+"-supersample-D.tif", super_disparity_exp);
          write_image(ostr.str()+"-transformed-R.tif", right_t);
          write_image(ostr.str()+"-delta-D.tif", disparity);
          write_image(ostr.str()+"-L.tif",
                      crop(left_pyramid[level], active_left_roi - left_roi[level].min()));
          std::cout << "Additive search: " << additive_search_range << std::endl;
          std::cout << "Kernel size: "<< m_kernel_size << std::endl;

          rl_disparity =
            calc_disparity(m_cost_type,
                           right_t,
                           crop(edge_extend(left_pyramid[level]),
                                active_left_roi - left_roi[level].min()
                                - additive_search_range.size()),
                           bounding_box(right_t),
                           additive_search_range.size(), m_kernel_size)
            - pixel_type(additive_search_range.size());

          stereo::cross_corr_consistency_check(disparity, rl_disparity,
                                               m_consistency_threshold, false);
          write_image(ostr.str()+"-delta-fltr-D.tif", disparity);
          write_image(ostr.str()+"-delta-rl-D.tif", rl_disparity);

          super_disparity =
            crop(super_disparity_exp,
                 BBox2i(-active_right_roi.min(),
                        -active_right_roi.min() + output_size)) +
            pixel_cast<PixelMask<Vector2f> >(disparity) +
            PixelMask<Vector2f>(additive_search_range.min());
          std::cout << "BBox of Disp: " << bounding_box(disparity) << std::endl;
          std::cout << "BBox of SDisp:" << bounding_box(super_disparity_exp) << std::endl;
          write_image(ostr.str()+"-improved-D.tif", super_disparity);
          /*{
            ImageView<PixelMask<Vector2f> > alternative =
              crop(super_disparity_exp,
                   BBox2i(Vector2i(16,16),
                          Vector2i(16,16) + output_size)) +
              pixel_cast<PixelMask<Vector2f> >(disparity);
            write_image(ostr.str()+"-improved-alt-D.tif",
                        alternative);
            ImageView<PixelMask<Vector2f> > alternative_smooth =
              block_rasterize(stereo::surface_fit(alternative),
                              surface_fit_tile, 2);
            copy_valid(alternative_smooth, alternative);
            blur_disparity(alternative_smooth,
                           BBox2i(Vector2i(),
                                  m_search_region.size() / scaling));
            write_image(ostr.str()+"-blurred-alt-D.tif", alternative_smooth);
            smooth_disparity = alternative_smooth;
            }*/

          smooth_disparity =
            block_rasterize(stereo::surface_fit(super_disparity),
                            surface_fit_tile, 2);
          copy_valid(smooth_disparity, super_disparity);
          blur_disparity(smooth_disparity,
                         BBox2i(Vector2i(),
                                m_search_region.size() / scaling));

          write_image(ostr.str()+"-blurred-D.tif", smooth_disparity);
        } // end of level loop

        std::cout << "Level 0 ------------------" << std::endl;

        BBox2i active_left_roi(Vector2i(), bbox_exp.size());
        active_left_roi.min() -= half_kernel;
        active_left_roi.max() += half_kernel;
        BBox2i active_right_roi = active_left_roi;
        active_right_roi.min() += additive_search_range.min();
        active_right_roi.max() += additive_search_range.max();
        std::cout << "Active L ROI: " << active_left_roi << std::endl;
        std::cout << "Active R ROI: " << active_right_roi << std::endl;

        super_disparity_exp =
          crop(edge_extend(2 * crop(resample(smooth_disparity, 2, 2),
                                    BBox2i(Vector2i(), bbox_exp.size()))),
               active_right_roi);

        right_t =
          crop(transform_no_edge
               (crop(edge_extend(right_pyramid[0]),
                     active_right_roi.min().x() - right_roi[0].min().x(),
                     active_right_roi.min().y() - right_roi[0].min().y(),
                     1, 1),
                stereo::DisparityTransform(super_disparity_exp)),
               active_right_roi - active_right_roi.min());

        // Hmm calc_disparity actually copies the imagery
        // again. Grr. There should be a speed up if I don't actually
        // raster the right image and just let calc disparity do it.
        //
        // Performing the final cross correlation between images. This
        // time however only processing the region we actually need
        // for output.
        BBox2i render_area_left = active_left_roi - active_left_roi.min();
        render_area_left.contract(m_padding);
        BBox2i render_area_right = bounding_box(right_t);
        render_area_right.contract(m_padding);
        disparity =
          calc_disparity(m_cost_type,
                         crop(left_pyramid[0], active_left_roi - left_roi[0].min()),
                         right_t, render_area_left,
                         additive_search_range.size(), m_kernel_size);
        rl_disparity =
          calc_disparity(m_cost_type,
                         right_t,
                         crop(edge_extend(left_pyramid[0]),
                              active_left_roi - left_roi[0].min()
                              - additive_search_range.size()),
                         render_area_right,
                         additive_search_range.size(), m_kernel_size)
          - pixel_type(additive_search_range.size());

        stereo::cross_corr_consistency_check(disparity, rl_disparity,
                                             m_consistency_threshold, false);
        write_image("level0-delta-D.tif", disparity);
        write_image("level0-supersample-D.tif", super_disparity_exp);
        BBox2i roi_super_disp(-active_right_roi.min().x() + m_padding,
                              -active_right_roi.min().y() + m_padding,
                              disparity.cols(), disparity.rows());
        std::cout << "ROI Super Disp: " << roi_super_disp << std::endl;
        super_disparity =
          crop(super_disparity_exp, roi_super_disp) +
          pixel_cast<PixelMask<Vector2f> >(disparity) +
          PixelMask<Vector2f>(additive_search_range.min());
        VW_ASSERT(super_disparity.cols() == bbox.width() &&
                  super_disparity.rows() == bbox.height(),
                  MathErr() << bounding_box(super_disparity) << " !fit in " << bbox_exp);

#if VW_DEBUG_LEVEL > 0
        watch.stop();
        double elapsed = watch.elapsed_seconds();
        vw_out(DebugMessage,"stereo") << "Tile " << bbox_exp << " processed in "
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
                               float consistency_threshold = 2,
                               int32 padding = 32) {
      typedef MappingPyramidCorrelationView<Image1T,Image2T,Mask1T,Mask2T,PreFilterT> result_type;
      return result_type( left.impl(), right.impl(), left_mask.impl(),
                          right_mask.impl(), filter.impl(), search_region,
                          kernel_size, cost_type,
                          consistency_threshold, padding );
    }

  }} // namespace vw::stereo

#endif//__VW_STEREO_MAPPING_PYRAMID_CORRELATION_VIEW_H__

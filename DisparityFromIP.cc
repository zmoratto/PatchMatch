#include <DisparityFromIP.h>

#include <vw/Image.h>
#include <vw/InterestPoint/InterestData.h>
#include <asp/Core/SoftwareRenderer.h>

#define REAL double
#define ANSI_DECLARATORS
#define VOID void
extern "C" {
#include "triangle.h"
}

using namespace vw;

struct XDisparity {
  inline float operator()(ip::InterestPoint const& ip1, ip::InterestPoint const& ip2) {
    return ip2.x - ip1.x;
  }
};

struct YDisparity {
  inline float operator()(ip::InterestPoint const& ip1, ip::InterestPoint const& ip2) {
    return ip2.y - ip1.y;
  }
};

template <class T>
void DrawTriangles( struct triangulateio const& input,
                    std::vector<ip::InterestPoint> const& ip1,
                    std::vector<ip::InterestPoint> const& ip2,
                    T value_functor,
                    stereo::SoftwareRenderer& renderer) {
  // Render triangles
  std::vector<float> vertices(6), intensities(3);
  //renderer.Clear(0);
  renderer.SetVertexPointer(2, &vertices[0]);
  renderer.SetColorPointer(1, &intensities[0]);
  for (int i = 0; i < input.numberoftriangles; i++ ) {
    for (int j = 0; j < 3; j++ ) {
      size_t ip_index = input.trianglelist[i * input.numberofcorners + j];
      vertices[j*2 + 0] = ip1[ip_index].x;
      vertices[j*2 + 1] = ip1[ip_index].y;
      intensities[j] = value_functor(ip1[ip_index], ip2[ip_index]);
    }
    renderer.DrawPolygon(0, 3);
  }
}

void DisparityFromIP(std::string const& match_filename,
                     vw::ImageView<vw::Vector2f> & output,
                     bool swap_order) {
  VW_ASSERT(output.cols() != 0 && output.rows() != 0,
            ArgumentErr() << "Output image must be allocated to the size desired for rendering");

  std::vector<ip::InterestPoint> ip1, ip2;
  read_binary_match_file(match_filename, ip1, ip2);
  if (swap_order) {
    std::swap(ip1, ip2);
  }

  // Build triangulateio objects to feed to triangle.c
  struct triangulateio in, out;
  in.numberofpoints = ip1.size();
  in.numberofpointattributes = 0;
  in.pointlist = (REAL*) malloc(in.numberofpoints * 2 * sizeof(REAL));
  for ( size_t i = 0; i < ip1.size(); i++ ) {
    in.pointlist[2 * i    ] = ip1[i].x;
    in.pointlist[2 * i + 1] = ip1[i].y;
  }
  in.pointattributelist = NULL;
  in.pointmarkerlist = NULL;
  in.pointattributelist = NULL;
  in.numberofsegments = 0;
  in.numberofholes = 0;
  in.numberofregions = 0;
  in.regionlist = NULL;
  out.trianglelist = NULL;
  triangulate("zNBP", &in, &out, (struct triangulateio*)NULL);
  std::cout << "number of triangles found: " << out.numberoftriangles << std::endl;

    // Create output buffer and Rasterizer 
  ImageView<float> x_disparity = copy(select_channel(output,0)),
    y_disparity = copy(select_channel(output,1));
  {
    stereo::SoftwareRenderer renderer(output.cols(), output.rows(),
                                      x_disparity.data());
    renderer.Ortho2D(0, output.cols(), 0, output.rows());
    DrawTriangles(out, ip1, ip2, XDisparity(), renderer);
  }
  {
    stereo::SoftwareRenderer renderer(output.cols(), output.rows(),
                                      y_disparity.data());
    renderer.Ortho2D(0, output.cols(), 0, output.rows());
    DrawTriangles(out, ip1, ip2, YDisparity(), renderer);
  }

  // Clean up triangle objects
  free(in.pointlist);
  free(out.trianglelist);

  // Put everything in the final image
  select_channel(output, 0) = x_disparity;
  select_channel(output, 1) = y_disparity;
}

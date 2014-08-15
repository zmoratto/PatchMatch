#ifndef __VW_STEREO_NELDERMEAD_H__
#define __VW_STEREO_NELDERMEAD_H__

#include <vw/Math/Vector.h>
#include <vw/Math/BBox.h>

namespace vw {
  namespace stereo {

    // For now, this is based on an implementation from Numerical Receipes
    template <int NDIM>
    struct Amoeba {
      const double ftol;
      int nfunc;  // The number of function evaluations
      int mpts;
      int ndim;
      double fmin;  // Function value at the minimum
      Vector<double, NDIM + 1> y;  // Function values at the vertices of the simplex
      Vector<double, NDIM> p[NDIM+1];  // Current simplex

      Amoeba(const double ftoll) : ftol(ftoll) {}
      // The constructor argument ftoll is the fractional convergence
      // tolerance to be achieved in the function value (n.b.!)
      template <class T>
      Vector<double, NDIM> minimize(Vector<double, NDIM> & point, const double del, T& func)
      // Multidimensional minimization of the function or functor
      // func(x), where x[0 .. ndim-1] is a vector in ndim dimensions,
      // by th edownhill simplex method of Nelder nd Mead. The initial
      // simplex is specified as in equation (10.5.1) by a
      // point[0..ndim-1] and a constant displacement del along each
      // coordinate direction. Returned is the location of the
      // minimum.
      {
        Vector<double, NDIM> dels;
        std::fill(dels.begin(), dels.end(), del);
        return minimize(point, dels, func);
      }

      template <class T>
      Vector<double, NDIM> minimize(Vector<double, NDIM> & point,
                                    Vector<double, NDIM> & dels, T &func)
      // Alternative interface that takes different displacements
      // dels[0..ndim-1] in different directions for the initial
      // simplex.
      {
        int ndim = point.size();
        Vector<double, NDIM> pp[NDIM+1];
        for (int i = 0; i < ndim + 1; i++) {
          pp[i] = point;
          if (i != 0) {
            pp[i][i-1] += dels[i-1];
          }
        }
        return minimize(pp, func);
      }

      template <class T>
      Vector<double, NDIM> minimize(Vector<double, NDIM> pp[], T & func)
      // Most general interface;: initial simplex specified by the
      // matrix pp[0..ndim][0..ndim-1]/ Its ndim+1 rows are
      // ndim-dimensional vectors that are the vertices of the
      // starting simplex.
      {
        const int NMAX=5000; // Maximum allowed number of function evaluations
        const double TINY=1e-10;
        int ihi, ilo, inhi;
        mpts = NDIM+1;
        ndim = NDIM;
        Vector<double, NDIM> psum, pmin, x;
        for (int i = 0; i < mpts; i++) {
          p[i] = pp[i];
        }
        for (int i = 0; i < mpts; i++) {
          x = p[i];
          y[i] = func(x);
        }
        nfunc = 0;
        get_psum(p, psum);
        for (;;) {
          ilo = 0;
          // First we must determine which point is te highest
          // (worst), next highest, and lowest (best), by looping over
          // the points in the simplex.
          if (y[0] > y[1])  {
            ihi = 0;
            inhi = 1;
          } else {
            ihi = 1;
            inhi = 0;
          }
          for (int i = 0; i < mpts; i++) {
            if (y[i] <= y[ilo]) ilo=i;
            if (y[i] > y[ihi]) {
              inhi = ihi;
              ihi = i;
            } else if (y[i] > y[inhi] && i != ihi) inhi = i;
          }
          double rtol = 2.0 * abs(y[ihi]-y[ilo])/(abs(y[ihi])+abs(y[ilo])+TINY);
          // Compute the fractional range from highest to lowest and
          // return if satisfactory
          if (rtol < ftol) { // If returning, put best point and value in slot 0
            std::swap(y[0], y[ilo]);
            for (int i = 0; i < ndim; i++ ) {
              std::swap(p[0][i], p[ilo][i]);
              pmin[i] = p[0][i];
            }
            fmin = y[0];
            return pmin;
          }
          if (nfunc >= NMAX) throw("NMAX exceeded");
          nfunc += 2;
          // Begin a new iteration. First extrapolate by a factor -1
          // through the face of the simplex across from the high
          // point, i.e., reflect the simplex from the high point.
          double ytry = amotry(p, y, psum, ihi, -1.0, func);
          if (ytry <= y[ilo]) {
            // Gives a result better than the best point, so try an
            // additional extrapolation by a factor 2.
            ytry = amotry(p, y, psum, ihi, 2.0, func);
          } else if (ytry >= y[ihi]) {
            // The reflected point is worse than the second-highest,
            // so look for an intermediate lower point, i.e., do a
            // one-dimensional contraction.
            double ysave = y[ihi];
            ytry = amotry(p, y, psum, ihi, 0.5, func);
            if (ytry >= ysave) { // Can't seem to get rid of that high point.
              for (int i = 0; i < mpts; i++) { // Better contract
                                               // around the lowest
                                               // (best) point
                if (i != ilo) {
                  p[i] = psum = 0.5 * (p[i] + p[ilo]);
                  y[i] = func(psum);
                }
              }
              nfunc += ndim; // Keep track of function evaluations.
              get_psum(p, psum); // Recompute psum.
            }
          } else {
            --nfunc; // Correct the evaluation count.
          }
        }
      }

      inline void get_psum(Vector<double, NDIM> p[], Vector<double, NDIM> &psum) {
        // Utility function.
        psum = Vector<double, NDIM>();
        for (int i = 0; i < mpts; i++) {
          psum += p[i];
        }
      }

      template <class T>
      double amotry(Vector<double, NDIM> p[], Vector<double, NDIM + 1> &y,
                    Vector<double, NDIM> &psum,
                    int ihi, double fac, T &func)
      // Helper function: Extrapolates by a factor fac through the
      // face of the simplex across from the high point, tries it, and
      // replaces the high point if the new point is better.
      {
        Vector<double, NDIM> ptry;
        double fac1 = (1.0 - fac) / ndim;
        double fac2 = fac1 - fac;
        ptry = psum * fac1 - p[ihi] * fac2;
        double ytry = func(ptry); // Evaluate the function at the trial point.
        if (ytry < y[ihi]) { // if it's better than the highest, then
                             // replace the highest.
          y[ihi] = ytry;
          psum += (ptry - p[ihi]);
          p[ihi] = ptry;
        }
        return ytry;
      }
    };

  }
}

#endif //__VW_STEREO_NELDERMEAD_H__

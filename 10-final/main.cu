#include <stdio.h>
#include "main.h"
#include <math.h>
#include <chrono>

__launch_bounds__(BLOCK_SIZE)
__global__ void dudt(
  Array3D<scalar,3> solution,
  Array3D<int,2> neighbours,
  Array3D<scalar,_NF> data,
  scalar g,
  scalar rho_0,
  Array3D<scalar,3> base,
  Array3D<scalar,3> out,
  scalar rhs_coeff,
  int n_triangles
){

  __shared__ SharedArray3D<3> xybath;
  int ie = blockIdx.x*blockDim.x + threadIdx.x;
  int ie_loc = threadIdx.x;

  if(ie < n_triangles){
    auto soli = load_elem(solution, ie);
    LocalArray3D<3> rhsi(0);

    for(int field = 0; field < 3; field++){
      for(int node = 0; node < 3; node++){
        xybath(field, node, ie_loc) = data(field, node, ie);
      }
    }

    // edge terms
    for (int side = 0; side <3; ++side) {
      int tri_l = ie;
      int closure_l = side;
      int tri_r     = neighbours(0, side, tri_l);
      int closure_r = neighbours(1, side, tri_l);

      scalar length,nx,ny;
      scalar x0 = xybath(_X, side,       ie_loc);
      scalar x1 = xybath(_X, (side+1)%3, ie_loc);
      scalar y0 = xybath(_Y, side,       ie_loc);
      scalar y1 = xybath(_Y, (side+1)%3, ie_loc);
      length = hypot(x0-x1,y0-y1);
      nx = -(y0-y1)/length;
      ny = (x0-x1)/length;

      for (int iq = 0; iq < gauss_edge_n; ++iq) {
        scalar phil[3];
        scalar xitl[2];
        fe_2d_closure_xi(gauss_edge_xi[iq],closure_l,xitl);
        tri_phi_e(xitl,phil);

        scalar soll[3];
        soll[0] = fe_2d_interp_field(soli, 0, 0, phil);
        soll[1] = fe_2d_interp_field(soli, 1, 0, phil);
        soll[2] = fe_2d_interp_field(soli, 2, 0, phil);

        scalar bathl = fe_2d_interp_field(xybath, _BATH, ie_loc, phil);
        scalar c_l   = sqrt(g*bathl + soll[0]);

        scalar flux[3] = {0};
        if (tri_r >= 0) {
          scalar phir[3];
          scalar xitr[2];
          fe_2d_closure_xi(gauss_edge_xi[iq],closure_r,xitr);
          tri_phi_e(xitr,phir);

          scalar solr[3];
          solr[0] = fe_2d_interp_field(solution, 0, tri_r, phir);
          solr[1] = fe_2d_interp_field(solution, 1, tri_r, phir);
          solr[2] = fe_2d_interp_field(solution, 2, tri_r, phir);

          scalar bathr = fe_2d_interp_field(data, _BATH, tri_r, phir);
          scalar c_r   = sqrt(g*bathr + solr[0]);

          iflux(c_l, c_r, soll, solr, bathl, bathr, nx, ny, g, flux);
        }

        scalar w = gauss_edge_w[iq]*(length/2);
        fe_2d_assemble_term(rhsi,3,3,phil,NULL,-w,flux);
      }
    }


    // volume term
    scalar dphi[3][2], jac;
    fe_2d_triangle(xybath,ie_loc,dphi,&jac);

    for (int iq=0; iq < gauss_tri_n; ++iq) {
      scalar phi[3];
      tri_phi(gauss_tri_xi[iq],phi);
      
      scalar sol[3];
      sol[0] = fe_2d_interp_field(soli, 0, 0, phi);
      sol[1] = fe_2d_interp_field(soli, 1, 0, phi);
      sol[2] = fe_2d_interp_field(soli, 2, 0, phi);

      scalar dsol[3][2];
      fe_2d_grad_field(soli, 0, 0, dphi, dsol[0]);
      fe_2d_grad_field(soli, 1, 0, dphi, dsol[1]);
      fe_2d_grad_field(soli, 2, 0, dphi, dsol[2]);

      scalar bath, cor, tau[2], gamma;
      bath   = fe_2d_interp_field(xybath, _BATH , ie_loc, phi);
      cor    = fe_2d_interp_field(data, _COR  , ie, phi);
      tau[0] = fe_2d_interp_field(data, _TAUX , ie, phi);
      tau[1] = fe_2d_interp_field(data, _TAUY , ie, phi);
      gamma  = fe_2d_interp_field(data, _GAMMA, ie, phi);
      
      scalar dbath[2];
      fe_2d_grad_field(xybath, _BATH, ie_loc, dphi, dbath);

      scalar s[9] = {0};
      fvolume(sol, dsol, bath, dbath, cor, tau, s, g, rho_0, gamma);

      scalar w = gauss_tri_w[iq]*jac;
      fe_2d_assemble_term(rhsi,3,3,phi,dphi,w,s);
    }

    // inverse mass matrix
    fe_2d_multiply_inv_mass_matrix(jac,rhsi);

    // write back
    if(base.data == solution.data){
      for(int field = 0; field < 3; field++){
        for(int node = 0; node < 3; node++){
          out(field, node, ie) = soli(field, node) + rhs_coeff*rhsi(field, node);
        }
      }
    } else {
      for(int field = 0; field < 3; field++){
        for(int node = 0; node < 3; node++){
          out(field, node, ie) = base(field, node, ie) + rhs_coeff*rhsi(field, node);
        }
      }      
    }
  }
}

__global__ void axpy(scalar* x, scalar* y, scalar* z, scalar a, scalar b, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n){
    z[i] = a*x[i] + b*y[i];
  }
}

int main(){
  Array3D<scalar,_NF> data;
  Array3D<int,2> neighbours;
  int n_elem = readmesh("square.txt", data, neighbours);
  int nelem_pad = AOSOA * ((n_elem+AOSOA-1)/AOSOA);

  Array3D<scalar,3> solution;
  solution.data = malloc_manged_flags<scalar>(3*3*nelem_pad);
  solution.n_elem = n_elem;

  Array3D<scalar,3> solution_mid;
  solution_mid.data = malloc_manged_flags<scalar>(3*3*nelem_pad);
  solution_mid.n_elem = n_elem;

  scalar dt = 2.0;
  int ngrid_dudt = (n_elem+BLOCK_SIZE-1)/BLOCK_SIZE;
  ERRCHK(cudaDeviceSynchronize());

  auto tstart = std::chrono::high_resolution_clock::now();
  const int niter = 10;
  double ts[niter+1] = {0};
  for(int iter = 0; iter < niter; iter++){
    int nsub = 1000;
    for(int i = 0; i < nsub; i++){
      // classical RK2
      dudt<<<ngrid_dudt,BLOCK_SIZE>>>(solution, neighbours, data, 9.81, 1000, solution, solution_mid, dt/2, n_elem);
      dudt<<<ngrid_dudt,BLOCK_SIZE>>>(solution_mid, neighbours, data, 9.81, 1000, solution, solution, dt, n_elem);
    }
    ERRCHK(cudaDeviceSynchronize());
    auto tstop = std::chrono::high_resolution_clock::now();
    ts[iter+1] = std::chrono::duration<double>(tstop-tstart).count()/nsub;
  }
  write_result("solution.txt", solution);
  double dt_min = ts[1];
  double dt_sum1 = 0;
  double dt_sum2 = 0;
  for(int i = 0; i < niter; i++){
    double dmt = ts[i+1]-ts[i];
    dt_sum1 += dmt;
    dt_sum2 += dmt*dmt;
    dt_min = fmin(dt_min, dmt);
  }
  double dt_mean = dt_sum1/niter;
  double dt_std = sqrt(dt_sum2/niter - dt_mean*dt_mean);
  printf("Min  time per iter: %.1f us\n", dt_min*1e6);
  printf("Mean time per iter: %.1f us\n", dt_mean*1e6);
  printf("Std : %.3f us\n", dt_std*1e6);
  return 0;
}

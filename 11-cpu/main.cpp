#include <stdio.h>
#include "main.h"
#include <math.h>
#include <chrono>

void dudt(
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

  ava_for<1>(0, n_triangles,  [=]__device__(v_index iie) {
    __shared__ SharedArray3D<3> xybath;
    v_index ie_loc = ie % BLOCK_SIZE;
    auto soli = load_elem(solution, ie);
    LocalArray3D<3> rhsi(0);

    for(int field = 0; field < 3; field++){
      for(int node = 0; node < 3; node++){
        xybath(field, node, ie_loc) = data(field, node, ie);
      }
    }

    // edge terms
    for (int side = 0; side <3; ++side) {
      v_index tri_l = ie;
      int closure_l = side;
      v_index tri_r     = neighbours(0, side, tri_l);
      v_index closure_r = neighbours(1, side, tri_l);

      v_scalar length,nx,ny;
      v_scalar x0 = xybath(_X, side,       ie_loc);
      v_scalar x1 = xybath(_X, (side+1)%3, ie_loc);
      v_scalar y0 = xybath(_Y, side,       ie_loc);
      v_scalar y1 = xybath(_Y, (side+1)%3, ie_loc);
      length = hypot(x0-x1,y0-y1);
      nx = -(y0-y1)/length;
      ny = (x0-x1)/length;

      for (int iq = 0; iq < gauss_edge_n; ++iq) {
        scalar phil[3];
        scalar xitl[2];
        fe_2d_closure_xi(gauss_edge_xi[iq],closure_l,xitl);
        tri_phi_e(xitl,phil);

        v_scalar soll[3];
        soll[0] = fe_2d_interp_field(soli, 0, 0, phil);
        soll[1] = fe_2d_interp_field(soli, 1, 0, phil);
        soll[2] = fe_2d_interp_field(soli, 2, 0, phil);

        v_scalar bathl = fe_2d_interp_field(xybath, _BATH, ie_loc, phil);
        v_scalar c_l   = sqrt(g*bathl + soll[0]);

        v_scalar flux[3] = {0};

        v_scalar phir[3];
        v_scalar xitr[2];
        fe_2d_closure_xi(gauss_edge_xi[iq],closure_r,xitr);
        tri_phi_e(xitr,phir);

        v_scalar solr[3];
        solr[0] = fe_2d_interp_field(solution, 0, tri_r, phir);
        solr[1] = fe_2d_interp_field(solution, 1, tri_r, phir);
        solr[2] = fe_2d_interp_field(solution, 2, tri_r, phir);

        v_scalar bathr = fe_2d_interp_field(data, _BATH, tri_r, phir);
        v_scalar c_r   = sqrt(g*bathr + solr[0]);

        iflux(c_l, c_r, soll, solr, bathl, bathr, nx, ny, g, flux);

        auto mask = tri_r < 0;
        if(any_of(mask)){
          for(int i = 0; i < AOSOA; i++){
            if(mask[i]){
              flux[0][i] = 0;
              flux[1][i] = 0;
              flux[2][i] = 0;
            }
          }
        }

        v_scalar w = gauss_edge_w[iq]*(length/2);
        fe_2d_assemble_term(rhsi,3,3,phil,NULL,-w,flux);
      }
    }


    // volume term
    v_scalar dphi[3][2], jac;
    fe_2d_triangle(xybath,ie_loc,dphi,jac);

    for (int iq=0; iq < gauss_tri_n; ++iq) {
      scalar phi[3];
      tri_phi(gauss_tri_xi[iq],phi);
      
      v_scalar sol[3];
      sol[0] = fe_2d_interp_field(soli, 0, 0, phi);
      sol[1] = fe_2d_interp_field(soli, 1, 0, phi);
      sol[2] = fe_2d_interp_field(soli, 2, 0, phi);

      v_scalar dsol[3][2];
      fe_2d_grad_field(soli, 0, 0, dphi, dsol[0]);
      fe_2d_grad_field(soli, 1, 0, dphi, dsol[1]);
      fe_2d_grad_field(soli, 2, 0, dphi, dsol[2]);

      v_scalar bath, cor, tau[2], gamma;
      bath   = fe_2d_interp_field(xybath, _BATH , ie_loc, phi);
      cor    = fe_2d_interp_field(data, _COR  , ie, phi);
      tau[0] = fe_2d_interp_field(data, _TAUX , ie, phi);
      tau[1] = fe_2d_interp_field(data, _TAUY , ie, phi);
      gamma  = fe_2d_interp_field(data, _GAMMA, ie, phi);
      
      v_scalar dbath[2];
      fe_2d_grad_field(xybath, _BATH, ie_loc, dphi, dbath);

      v_scalar s[9] = {0};
      fvolume(sol, dsol, bath, dbath, cor, tau, s, g, rho_0, gamma);

      v_scalar w = gauss_tri_w[iq]*jac;
      fe_2d_assemble_term(rhsi,3,3,phi,dphi,w,s);
    }

    // inverse mass matrix
    fe_2d_multiply_inv_mass_matrix(jac,rhsi);

    // write back
    if(base.data == solution.data){
      for(int field = 0; field < 3; field++){
        for(int node = 0; node < 3; node++){
          out.store(field, node, ie, soli(field, node) + rhs_coeff*rhsi(field, node));
          // out(field, node, ie) = soli(field, node) + rhs_coeff*rhsi(field, node)
        }
      }
    } else {
      for(int field = 0; field < 3; field++){
        for(int node = 0; node < 3; node++){
          // out(field, node, ie) = base(field, node, ie) + rhs_coeff*rhsi(field, node);
          out.store(field, node, ie, base(field, node, ie) + rhs_coeff*rhsi(field, node));
        }
      }      
    }
  });
}


int main(int argc, char** argv){
  int niter = 10;
  int nsub = 2;
  for(int i = 0; i < argc; i++){
    if(strcmp(argv[i], "--profile") == 0){
      niter = 1;
      nsub = 2;
    }
  }
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
  // ERRCHK(cudaDeviceSynchronize());

  auto tstart = std::chrono::high_resolution_clock::now();
  double *ts = new double[niter+1];
  for(int iter = 0; iter < niter; iter++){
    for(int i = 0; i < nsub; i++){
      // classical RK2
      dudt(solution, neighbours, data, 9.81, 1000, solution, solution_mid, dt/2, n_elem);
      dudt(solution_mid, neighbours, data, 9.81, 1000, solution, solution, dt, n_elem);
    }
    // ERRCHK(cudaDeviceSynchronize());
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
  delete[] ts;
  return 0;
}

// advisor command for roofline analysis w floats
// advisor --collect=survey --project-dir=./advi -- ./main --profile
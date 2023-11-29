#include <stdio.h>
#include "main.h"
#include <math.h>
#include <chrono>

enum {_X=0,_Y,_BATH,_GAMMA,_COR,_TAUX,_TAUY,_C,_NF};

template <typename T>
T* malloc_manged_flags(size_t count){
  T* ptr;
  ERRCHK(cudaMallocManaged(&ptr, count*sizeof(T)));
  ERRCHK(cudaMemAdvise(ptr, count, cudaMemAdviseSetPreferredLocation, cudaMemLocationTypeDevice));
  ERRCHK(cudaMemAdvise(ptr, count, cudaMemAdviseSetAccessedBy, cudaMemLocationTypeDevice));
  ERRCHK(cudaMemset(ptr, 0, count*sizeof(T)));
  return ptr;
}


int readmesh(const char* filename, Array3D<scalar,_NF> &data, Array3D<int,2> &neighbours){
  FILE *fp = fopen("square.txt","r");
  int n_elem;
  fscanf(fp,"%d",&n_elem);
  data.data = malloc_manged_flags<scalar>(_NF*3*n_elem);
  data.n_elem = n_elem;
  neighbours.data = malloc_manged_flags<int>(2*3*n_elem);
  neighbours.n_elem = n_elem;

  for(int elem = 0; elem < n_elem; elem++){
    for(int node = 0; node < 3; node++){
      double x=NAN, y=NAN, bath=NAN, gamma=NAN, cor=NAN, taux=NAN, tauy=NAN, c=NAN;
      int n=2147483647, cl=2147483647;
      fscanf(fp,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%d,%d",&x,&y,&bath,&gamma,&cor,&taux,&tauy,&c,&n,&cl);
      data(_X,node,elem) = x;
      data(_Y,node,elem) = y;
      data(_BATH,node,elem) = bath;
      data(_GAMMA,node,elem) = gamma;
      data(_COR,node,elem) = cor;
      data(_TAUX,node,elem) = taux;
      data(_TAUY,node,elem) = tauy;
      data(_C,node,elem) = c;
      neighbours(0,node,elem) = n;
      neighbours(1,node,elem) = cl;
    }
  }
  fclose(fp);
  return n_elem;
}

void write_result(const char* filename, Array3D<scalar,3> &solution){
  FILE *fp = fopen(filename,"w");
  int n_elem = solution.n_elem;
  for(int elem = 0; elem < n_elem; elem++){
    for(int node = 0; node < 3; node++){
      fprintf(fp,"%e,%e,%e\n", solution(0,node,elem), solution(1,node,elem), solution(2,node,elem));
    }
  }
  fclose(fp);
}


__device__ inline static scalar det2x2(const scalar mat[2][2]) {
  return mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0];
}

__device__ inline static scalar inv2x2(const scalar mat[2][2], scalar inv[2][2])
{
  scalar det = det2x2(mat);
  scalar ud = 1.f / det;
  inv[0][0] =  mat[1][1] * ud;
  inv[1][0] = -mat[1][0] * ud;
  inv[0][1] = -mat[0][1] * ud;
  inv[1][1] =  mat[0][0] * ud;
  return det;
}

template<typename Array>
__device__ void inline static fe_2d_triangle(Array &xy, int elem, scalar dphi[3][2], scalar *jac) {
  scalar dxdxi[2][2]={{0}};
  scalar dxidx[2][2];
  for (int node = 0; node < 3; node++) {
    dxdxi[0][0] += xy(0, node, elem)*tri_dphi_dxi[node][0];
    dxdxi[0][1] += xy(0, node, elem)*tri_dphi_dxi[node][1];
    dxdxi[1][0] += xy(1, node, elem)*tri_dphi_dxi[node][0];
    dxdxi[1][1] += xy(1, node, elem)*tri_dphi_dxi[node][1];
  }
  *jac = inv2x2(dxdxi,dxidx);
  for (int node = 0; node < 3; node++) {
    for (int j = 0; j < 2; ++j) {
      dphi[node][j] = tri_dphi_dxi[node][0]*dxidx[0][j] + tri_dphi_dxi[node][1]*dxidx[1][j];
    }
  } 
}

template<typename Array>
__device__ inline static scalar fe_2d_interp_field(Array &solution, int field, int ie, const scalar phi[3]){
  scalar v = 0;
  for (int node = 0; node < 3; ++node) {
    v += solution(field, node, ie)*phi[node];
  }
  return v;
}

template<typename Array>
__device__ inline static void fe_2d_assemble_term(Array &rhse,int nphi, int nf, scalar *phi, scalar dphi[3][2], scalar w, scalar *s) {
  for (int node = 0; node < nphi; node++) { 
    for (int field = 0; field < nf; ++field) {
      scalar r = phi[node]*s[field];
      if(dphi) 
        r += dphi[node][0]*s[nf+field*2+0]+dphi[node][1]*s[nf+field*2+1];
      rhse(field, node, 0) += w*r;
    }
  }
}

__device__ inline static void tri_phi(const scalar pts[2], scalar *phi) {
  scalar xi = pts[0], eta = pts[1];
  phi[0] = 1-xi-eta;
  phi[1] = xi;
  phi[2] = eta;
}

__device__ inline static void tri_phi_e(scalar pts[2], scalar *phi) {
  scalar xi = pts[0], eta = pts[1];
  phi[0] = 1-xi-eta;
  phi[1] = xi;
  phi[2] = eta;
}

template<typename Array>
__device__ inline static void fe_2d_grad_field(Array &solution, int field, int ie, const scalar dphi[3][2], scalar v[2]){
  for (int k = 0; k < 2; ++k){
    v[k] = 0;
  }
  for (int node = 0; node < 3; ++node) {
    for (int k = 0; k < 2; ++k){
      v[k] += solution(field, node, ie)*dphi[node][k];
    }
  }
}

__device__ inline static void fe_2d_multiply_inv_mass_matrix(scalar jac, LocalArray3D<3> &rhse) {
  //multiply by inv mass matrix
  for (int ifield = 0; ifield < 3; ++ifield) {
    scalar rhsl[3];
    for (int inode = 0; inode <3; ++inode) {
      rhsl[inode] = rhse(ifield, inode);
    }
    for (int inode = 0; inode <3; ++inode) {
      rhse(ifield, inode) = (-6*(rhsl[0]+rhsl[1]+rhsl[2])+24*rhsl[inode])/jac;
    }
  }
}

__global__ void dudt(
  Array3D<scalar,3> solution,
  Array3D<int,2> neighbours,
  Array3D<scalar,_NF> data,
  scalar g,
  scalar rho_0,
  Array3D<scalar,3> rhs,
  int n_triangles
){

  int ie = blockIdx.x*blockDim.x + threadIdx.x;
  // int ie_loc = threadIdx.x;

  if(ie < n_triangles){
    auto soli = load_elem(solution, ie);
    LocalArray3D<3> rhsi(0);

    // edge terms
    for (int side = 0; side <3; ++side) {
      int tri_l = ie;
      int closure_l = side;
      int tri_r     = neighbours(0, side, tri_l);
      int closure_r = neighbours(1, side, tri_l);

      scalar length,nx,ny;
      scalar x0 = data(_X, side,       tri_l);
      scalar x1 = data(_X, (side+1)%3, tri_l);
      scalar y0 = data(_Y, side,       tri_l);
      scalar y1 = data(_Y, (side+1)%3, tri_l);
      length = hypot(x0-x1,y0-y1);
      nx = -(y0-y1)/length;
      ny = (x0-x1)/length;

      
      LocalArray3D<3> solri;
      if (tri_r >= 0) {
        solri = load_elem(solution, tri_r);
      }
      
      for (int iq = 0; iq < gauss_edge_n; ++iq) {
        scalar phil[3];
        scalar xitl[2];
        fe_2d_closure_xi(gauss_edge_xi[iq],closure_l,xitl);
        tri_phi_e(xitl,phil);

        scalar soll[3];
        soll[0] = fe_2d_interp_field(soli, 0, 0, phil);
        soll[1] = fe_2d_interp_field(soli, 1, 0, phil);
        soll[2] = fe_2d_interp_field(soli, 2, 0, phil);

        scalar bathl = fe_2d_interp_field(data, _BATH, tri_l, phil);
        scalar c_l   = fe_2d_interp_field(data, _C   , tri_l, phil);

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
          scalar c_r   = fe_2d_interp_field(data, _C   , tri_r, phir);

          iflux(c_l, c_r, soll, solr, bathl, bathr, nx, ny, g, flux);
        }

        scalar w = gauss_edge_w[iq]*(length/2);
        fe_2d_assemble_term(rhsi,3,3,phil,NULL,-w,flux);
      }
    }


    // volume term
    scalar dphi[3][2], jac;
    fe_2d_triangle(data,ie,dphi,&jac);

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
      bath   = fe_2d_interp_field(data, _BATH , ie, phi);
      cor    = fe_2d_interp_field(data, _COR  , ie, phi);
      tau[0] = fe_2d_interp_field(data, _TAUX , ie, phi);
      tau[1] = fe_2d_interp_field(data, _TAUY , ie, phi);
      gamma  = fe_2d_interp_field(data, _GAMMA, ie, phi);
      
      scalar dbath[2];
      fe_2d_grad_field(data, _BATH, ie, dphi, dbath);

      scalar s[9] = {0.f};
      fvolume(sol, dsol, bath, dbath, cor, tau, s, g, rho_0, gamma);

      scalar w = gauss_tri_w[iq]*jac;
      fe_2d_assemble_term(rhsi,3,3,phi,dphi,w,s);

    }

    // inverse mass matrix
    fe_2d_multiply_inv_mass_matrix(jac,rhsi);

    // write back
    for (int field = 0; field < 3; ++field) {
      for (int node = 0; node < 3; ++node) {
        rhs(field, node, ie) = rhsi(field, node);
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

  Array3D<scalar,3> solution;
  solution.data = malloc_manged_flags<scalar>(3*3*n_elem);
  solution.n_elem = n_elem;

  Array3D<scalar,3> solution_mid;
  solution_mid.data = malloc_manged_flags<scalar>(3*3*n_elem);
  solution_mid.n_elem = n_elem;

  Array3D<scalar,3> rhs;
  rhs.data = malloc_manged_flags<scalar>(3*3*n_elem);
  rhs.n_elem = n_elem;  

  scalar dt = 2.0;
  int ngrid_dudt = (n_elem+BLOCK_SIZE-1)/BLOCK_SIZE;
  int ngrid_axpy = (n_elem*9+BLOCK_SIZE-1)/BLOCK_SIZE;
  ERRCHK(cudaDeviceSynchronize());

  auto tstart = std::chrono::high_resolution_clock::now();
  const int niter = 10;
  double ts[niter+1] = {0};
  for(int iter = 0; iter < niter; iter++){
    int nsub = 1000;
    for(int i = 0; i < nsub; i++){
      // classical Explicit Euler
      // dudt<<<ngrid_dudt,BLOCK_SIZE>>>(solution, neighbours, data, 9.81, 1000, rhs, n_elem);
      // axpy<<<ngrid_axpy,BLOCK_SIZE>>>(solution.data, rhs.data, solution.data, 1, dt, n_elem*9);

      // classical RK2
      dudt<<<ngrid_dudt,BLOCK_SIZE>>>(solution, neighbours, data, 9.81, 1000, rhs, n_elem);
      axpy<<<ngrid_axpy,BLOCK_SIZE>>>(solution.data, rhs.data, solution_mid.data, 1, dt/2, n_elem*9);
      dudt<<<ngrid_dudt,BLOCK_SIZE>>>(solution_mid, neighbours, data, 9.81, 1000, rhs, n_elem);
      axpy<<<ngrid_axpy,BLOCK_SIZE>>>(solution.data, rhs.data, solution.data, 1, dt, n_elem*9);
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
  printf("Min time per iter : %.1f us\n", dt_min*1e6);
  printf("Mean time per iter: %.1f us\n", dt_mean*1e6);
  printf("Std : %.3f us\n", dt_std*1e6);
  return 0;
}


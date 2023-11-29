#include <stdio.h>
#include <cuda_runtime.h>
typedef float scalar;
#define BLOCK_SIZE 128

template<typename T, int n_fields>
class Array3D {
  public:
  T *data;
  int n_elem;
  __host__ __device__ constexpr static int num_fields(){return n_fields;};
  __host__ __device__ inline T &operator()(int field, int node, int elem) {
    // return data[n_elem*3*field + n_elem*node + elem];
    return data[elem*(n_fields*3) + node*n_fields + field];
  }
};

template<int n_fields>
class SharedArray3D {
  public:
  scalar data[BLOCK_SIZE*n_fields*3];
  __device__ constexpr static int num_fields(){return n_fields;};
  __device__ inline scalar &operator()(int field, int node, int elem) {
    return data[BLOCK_SIZE*3*field + BLOCK_SIZE*node + elem];
  }
};

#define ERRCHK(err) (errchk(err, __FILE__, __LINE__ ))
static inline void errchk(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "\n\n%s in %s:%d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

template<int n_fields>
class LocalArray3D {
  public:
  scalar data[n_fields*3];
  __device__ constexpr static int num_fields(){return n_fields;};
  __device__ inline scalar &operator()(int field, int node, int elem) {
    return data[3*field + node];
  }
  __device__ inline scalar &operator()(int field, int node) {
    return data[3*field + node];
  }
  __device__ LocalArray3D(scalar v = 0) {
    for(int i = 0; i < n_fields*3; i++) data[i] = v;
  }
};

template<int n_fields>
__device__ LocalArray3D<n_fields> load_elem(Array3D<scalar, n_fields> &array, int elem) {
  LocalArray3D<n_fields> local;
  for(int field = 0; field < n_fields; field++) {
    for(int node = 0; node < 3; node++) {
      local(field, node) = array(field, node, elem);
    }
  }
  return local;
}

template<int n_fields>
__device__ void store_elem(Array3D<scalar, n_fields> &array, int elem, LocalArray3D<n_fields> &local) {
  for(int field = 0; field < n_fields; field++) {
    for(int node = 0; node < 3; node++) {
      array(field, node, elem) = local(field, node);
    }
  }
}




#define gauss_tri_n 6
__device__ __constant__ static const scalar gauss_tri_xi[gauss_tri_n][2] = {
  {0.659027622374092, 0.231933368553031},
  {0.659027622374092, 0.109039009072877},
  {0.231933368553031, 0.659027622374092},
  {0.231933368553031, 0.109039009072877},
  {0.109039009072877, 0.659027622374092},
  {0.109039009072877, 0.231933368553031}};
__device__ __constant__ static const scalar gauss_tri_w[gauss_tri_n] = {1./12,1./12,1./12,1./12,1./12,1./12};

#define gauss_edge_n 2
__device__ const scalar gauss_edge_xi[2] = { -0.5773502691896257, 0.5773502691896257};
__device__ const scalar gauss_edge_w[2] = {1.0,1.0};

__device__ __constant__ static const int closure_cl[6][4] = {{0,1,4,3}, {1,2,5,4}, {2,0,3,5},{1,0,3,4}, {2,1,4,5}, {0,2,5,3}};
__device__ __constant__ static const scalar tri_dphi_dxi[3][2] = {{-1,-1},{1,0},{0,1}};

__device__ inline static void fe_2d_closure_xi(scalar xie, int cl, scalar xit[2]) {
  scalar x = (1+xie)/2.0;
  switch (cl) {
    case 0 : xit[0]=  x; xit[1]=  0; break;
    case 1 : xit[0]=1-x; xit[1]=  x; break;
    case 2 : xit[0]=  0; xit[1]=1-x; break;
    case 3 : xit[0]=1-x; xit[1]=  0; break;
    case 4 : xit[0]=  x; xit[1]=1-x; break;
    case 5 : xit[0]=  0; xit[1]=  x; break;
    default: printf("fe_2d_closure_xi: cl %d\n", cl);
  }
}

__device__ void inline static iflux(scalar c_l, scalar c_r, scalar soll[3], scalar solr[3], scalar bath_l, scalar bath_r, scalar nx, scalar ny, scalar g, scalar flux[3]) {
  const scalar tx = -ny, ty = nx;
  scalar Hunl = nx*soll[1]+ny*soll[2];
  scalar Hutl = tx*soll[1]+ty*soll[2];
  scalar Hunr = nx*solr[1]+ny*solr[2];
  scalar Hutr = tx*solr[1]+ty*solr[2];
  scalar etal = soll[0];
  scalar etar = solr[0];
  scalar Hl = etal+bath_l;
  scalar Hr = etar+bath_r;

  scalar Hun = (Hunl+Hunr)/2.0;
  scalar c = fmax(c_l,c_r);

  scalar flux_H = 0.0;
  scalar flux_n = c*(Hunl-Hunr)/2.0;
  scalar flux_t = 0.0;

  scalar umaxl = sqrt(g*Hl);
  scalar umaxr = sqrt(g*Hr);
  scalar unl = Hunl/Hl, unr = Hunr/Hr;

  // advection
  scalar un = (unl+unr)/2.0;
  flux_t += un*(un > 0 ? Hutl : Hutr);
  flux_n += 0.5*(Hunl*unl + Hunr*unr);

  // gravity
  flux_H += Hun + c*(etal-etar)/2.0;
  flux_n -= g*(bath_l + (etal+etar)/2.0)*(etal-etar)/2.0;

  scalar flux_u = flux_n*nx+flux_t*tx;
  scalar flux_v = flux_n*ny+flux_t*ty;

  flux[0] = flux_H;
  flux[1] = flux_u;
  flux[2] = flux_v;
}

__device__ inline static void fvolume(scalar sol[3], scalar dsol[3][2], scalar bath, scalar dbath[2], scalar cor, scalar tau[2], scalar s[9], scalar g, scalar rho_0, scalar gamma) {
  for (int i = 0; i<9; ++i) s[i] = 0;
  scalar *f = s+3;

  scalar H = sol[0]+bath;
  scalar *Hu = sol+1;

  // wave equation
  s[1] += - g*H*dsol[0][0]; // - datm_press[0]/rho_0;
  s[2] += - g*H*dsol[0][1]; // - datm_press[1]/rho_0;
  f[0*2+0] += Hu[0];
  f[0*2+1] += Hu[1];

  scalar u[2] = {Hu[0]/H, Hu[1]/H};
  scalar normu = hypot(u[0], u[1]);

  s[1] += tau[0]/rho_0+cor*Hu[1];
  s[2] += tau[1]/rho_0-cor*Hu[0];

  s[1] -= gamma*Hu[0] * normu;
  s[2] -= gamma*Hu[1] * normu;
  f[1*2+0] += u[0]*Hu[0];
  f[1*2+1] += u[0]*Hu[1];
  f[2*2+0] += u[1]*Hu[0];
  f[2*2+1] += u[1]*Hu[1];
}

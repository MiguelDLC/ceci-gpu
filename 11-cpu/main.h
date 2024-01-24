#ifndef MAIN_H
#define MAIN_H
#include <stdio.h>
#include <cmath>
#include <cstring>
#include <experimental/simd>
using namespace std::experimental::parallelism_v2;

#define ILP

#ifdef __CUDACC__
#include <cuda_runtime.h>
/////////// CUDA error checking, no changes needed ///////////
#define ERRCHK(err) (errchk(err, __FILE__, __LINE__ ))
static inline void errchk(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "\n\n%s in %s:%d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#else

#define __shared__ 
#define __host__
#define __device__
#define __global__
#define __constant__
// #define __noinline__ __attribute__((noinline))

// gpu structs : don't care
#define gpuStream_t void*
#define gpuDeviceProp void*
#define gpuEvent_t void*

// gpu constants : don't care
typedef enum {
  gpu_memcpy_host_to_host = 0,
  gpu_memcpy_host_to_device = 1,
  gpu_memcpy_device_to_host = 2,
  gpu_memcpy_device_to_device = 3,
  gpu_memcpy_default = 4
} gpuMemcpyKind;

typedef enum {
  gpu_stream_default = 0,
  gpu_stream_non_blocking = 1
} gpuStreamKind;

#define gpu_host_mallocDefault 0
#define gpu_event_disable_timing 1

// gpu functions : the ones we don't care about
#define gpu_get_device_count gpu_success
#define gpu_get_device_properties gpu_success
#define gpu_set_device gpu_success
#define gpu_device_get_stream_priority_range gpu_success
#define gpu_stream_create_with_priority gpu_success
#define gpu_event_create_with_flags(event, flags) gpu_success
#define gpu_event_record(event, stream) gpu_success
#define gpu_stream_wait_event(stream, event, flags) gpu_success
#define gpu_event_synchronize(gpuEvent_t) gpu_success
#define gpu_event_destroy(gpuEvent_t) gpu_success
#define gpu_stream_synchronize(stream) gpu_success
#define gpu_device_synchronise() gpu_success
typedef enum {
  gpu_success = 0,
  gpu_failure = 1
} gpuError;

#endif


typedef float scalar;
#define BLOCK_SIZE 1
#define AOSOA 32
using v_index = fixed_size_simd<int, AOSOA>; 
using v_scalar = fixed_size_simd<scalar, AOSOA>;
using v_mask = simd_mask<scalar, simd_abi::_Fixed<AOSOA>>;

const char* meshfile = "/../data/small/square.txt";


enum {_X=0,_Y,_BATH,_GAMMA,_COR,_TAUX,_TAUY,_C,_NF};
template<typename T, int n_fields>
class Array3D {
  public:
  T *data;
  int n_elem;
  __host__ __device__ constexpr static int num_fields(){return n_fields;};
  __host__ __device__ inline T &operator()(int field, int node, int elem) {
    int bloc = elem / AOSOA;
    int rem  = elem % AOSOA;
    return data[bloc*AOSOA*n_fields*3 + AOSOA*3*field + AOSOA*node + rem];
  }
  __host__ __device__ inline T &operator()(int field, int node, int elem) const {
    int bloc = elem / AOSOA;
    int rem  = elem % AOSOA;
    return data[bloc*AOSOA*n_fields*3 + AOSOA*3*field + AOSOA*node + rem];
  }

  __host__ __device__ inline fixed_size_simd<T,AOSOA> operator()(int field, int node, const v_index &elem) const {
    auto bloc = elem / AOSOA;
    auto rem  = elem % AOSOA;
    fixed_size_simd<T,AOSOA> res;
    for(int i = 0; i < AOSOA; i++) {
      res[i] = (elem[i] < 0) ? 0 : data[bloc[i]*AOSOA*n_fields*3 + AOSOA*3*field + AOSOA*node + rem[i]];
    }
    return res;
  }

  __host__ __device__ inline void store(int field, int node, const v_index &elem, const fixed_size_simd<T,AOSOA> &data) const {
    int el0 = elem[0];
    int bloc = el0 / AOSOA;
    T* ptr = &this->data[bloc*AOSOA*n_fields*3 + AOSOA*3*field + AOSOA*node];
    data.copy_to(ptr, vector_aligned);
    return ;
  }

};

// good class, no changes needed
template<int n_fields>
class LocalArray3D {
  public:
  v_scalar data[n_fields*3];
  __device__ constexpr static int num_fields(){return n_fields;};
  __device__ inline v_scalar &operator()(int field, int node, int elem) {
    return data[3*field + node];
  }
  __device__ inline v_scalar &operator()(int field, int node) {
    return data[3*field + node];
  }
  __device__ LocalArray3D(v_scalar v = 0) {
    for(int i = 0; i < n_fields*3; i++) data[i] = v;
  }
};

__device__ inline static void fe_2d_closure_xi(scalar xie, v_index closure, v_scalar xit[2]) {
  scalar x = (1+xie)/2;
  for(int i = 0; i < AOSOA; i++) {
    int cl = closure[i];
    switch (cl) {
      case 0 : xit[0][i]=  x; xit[1][i]=  0; break;
      case 1 : xit[0][i]=1-x; xit[1][i]=  x; break;
      case 2 : xit[0][i]=  0; xit[1][i]=1-x; break;
      case 3 : xit[0][i]=1-x; xit[1][i]=  0; break;
      case 4 : xit[0][i]=  x; xit[1][i]=1-x; break;
      case 5 : xit[0][i]=  0; xit[1][i]=  x; break;
    }
  }
}


__device__ inline static void fe_2d_closure_xi(scalar xie, int cl, scalar xit[2]) {
  scalar x = (1+xie)/2;
  switch (cl) {
    case 0 : xit[0]=  x; xit[1]=  0; break;
    case 1 : xit[0]=1-x; xit[1]=  x; break;
    case 2 : xit[0]=  0; xit[1]=1-x; break;
    case 3 : xit[0]=1-x; xit[1]=  0; break;
    case 4 : xit[0]=  x; xit[1]=1-x; break;
    case 5 : xit[0]=  0; xit[1]=  x; break;
  }
}

__device__ inline static v_scalar det2x2(const v_scalar mat[2][2]) {
  return mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0];
}

__device__ inline static v_scalar inv2x2(const v_scalar mat[2][2], v_scalar inv[2][2]){
  v_scalar det = det2x2(mat);
  v_scalar ud = 1 / det;
  inv[0][0] =  mat[1][1] * ud;
  inv[1][0] = -mat[1][0] * ud;
  inv[0][1] = -mat[0][1] * ud;
  inv[1][1] =  mat[0][0] * ud;
  return det;
}

__device__ inline static void fe_2d_multiply_inv_mass_matrix(const v_scalar &jac, LocalArray3D<3> &rhse) {
  //multiply by inv mass matrix
  for (int ifield = 0; ifield <3; ++ifield) {
    v_scalar rhsl[3];
    for (int inode = 0; inode <3; ++inode) {
      rhsl[inode] = rhse(ifield, inode);
    }
    for (int inode = 0; inode <3; ++inode) {
      rhse(ifield, inode) = (-6*(rhsl[0]+rhsl[1]+rhsl[2])+24*rhsl[inode])/jac;
    }
  }
}

__host__ __device__ inline v_scalar simd_select(const v_mask &mask, const v_scalar &a, const v_scalar &b) {
  v_scalar res;
  for(int i = 0; i < AOSOA; i++) {
    res[i] = mask[i] ? a[i] : b[i];
  }
  return res;
}

__device__ void inline static iflux(v_scalar &c_l, v_scalar &c_r, v_scalar soll[3], v_scalar solr[3], v_scalar &bath_l, v_scalar &bath_r, v_scalar &nx, v_scalar &ny, scalar g, v_scalar flux[3]) {
  const v_scalar tx = -ny, ty = nx;
  v_scalar Hunl = nx*soll[1]+ny*soll[2];
  v_scalar Hutl = tx*soll[1]+ty*soll[2];
  v_scalar Hunr = nx*solr[1]+ny*solr[2];
  v_scalar Hutr = tx*solr[1]+ty*solr[2];
  v_scalar etal = soll[0];
  v_scalar etar = solr[0];
  v_scalar Hl = etal+bath_l;
  v_scalar Hr = etar+bath_r;

  v_scalar Hun = (Hunl+Hunr)/2;
  v_scalar c = fmax(c_l,c_r);

  v_scalar flux_H = 0;
  v_scalar flux_n = c*(Hunl-Hunr)/2;
  v_scalar flux_t = 0;

  v_scalar umaxl = sqrt(g*Hl);
  v_scalar umaxr = sqrt(g*Hr);
  v_scalar unl = Hunl/Hl, unr = Hunr/Hr;

  // advection
  v_scalar un = (unl+unr)/2;
  flux_t += un * simd_select(un > 0, Hutl, Hutr);

  flux_n += (Hunl*unl + Hunr*unr)/2;

  // gravity
  flux_H += Hun + c*(etal-etar)/2;
  flux_n -= g*(bath_l + (etal+etar)/2)*(etal-etar)/2;

  v_scalar flux_u = flux_n*nx+flux_t*tx;
  v_scalar flux_v = flux_n*ny+flux_t*ty;

  flux[0] = flux_H;
  flux[1] = flux_u;
  flux[2] = flux_v;
}

/////////// END OF THE EXERCISE, NO CHANGES NEEDED BELOW ///////////

// good function, no changes needed
__device__ inline static void fvolume(v_scalar sol[3], v_scalar dsol[3][2], v_scalar bath, v_scalar dbath[2], v_scalar cor, v_scalar tau[2], v_scalar s[9], scalar g, scalar rho_0, v_scalar gamma) {
  for (int i = 0; i<9; ++i) s[i] = 0;
  v_scalar *f = s+3;

  v_scalar H = sol[0]+bath;
  v_scalar *Hu = sol+1;

  // wave equation
  s[1] += - g*H*dsol[0][0];
  s[2] += - g*H*dsol[0][1];
  f[0*2+0] += Hu[0];
  f[0*2+1] += Hu[1];

  v_scalar u[2] = {Hu[0]/H, Hu[1]/H};
  v_scalar normu = hypot(u[0], u[1]);

  s[1] += tau[0]/rho_0+cor*Hu[1];
  s[2] += tau[1]/rho_0-cor*Hu[0];

  s[1] -= gamma*Hu[0] * normu;
  s[2] -= gamma*Hu[1] * normu;
  f[1*2+0] += u[0]*Hu[0];
  f[1*2+1] += u[0]*Hu[1];
  f[2*2+0] += u[1]*Hu[0];
  f[2*2+1] += u[1]*Hu[1];
}


#define ERRCHK(err) (errchk(err, __FILE__, __LINE__ ))
#define CCLCHCK(cmd) cmd
static inline void errchk(gpuError err, const char *file, int line) {
  if (err != gpu_success) {
    fprintf(stderr, "\n\n%s in %s:%d\n", "Unknown error", file, line);
    exit(EXIT_FAILURE);
  }
}

/////////// Gauss quadrature, no changes needed ///////////
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


/////////// Helper functions, no changes needed ///////////
// template <typename T>
// static T* malloc_manged_flags(size_t count){
//   T* ptr;
//   ERRCHK(cudaMallocManaged(&ptr, count*sizeof(T)));
//   ERRCHK(cudaMemAdvise(ptr, count, cudaMemAdviseSetPreferredLocation, 0));
//   ERRCHK(cudaMemAdvise(ptr, count, cudaMemAdviseSetAccessedBy, 0));
//   ERRCHK(cudaMemset(ptr, 0, count*sizeof(T)));
//   return ptr;
// }

template <typename T>
static T* malloc_manged_flags(size_t count){
  constexpr size_t align = AOSOA*sizeof(T);
  count = align*((count+align-1)/align);
  T* ptr = (T*) aligned_alloc(align, count*sizeof(T));
  memset(ptr, 0, count*sizeof(T));
  return ptr;
}



#include <iostream>
#include <limits.h>
#include <unistd.h>

std::string getExecutablePath() {
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len != -1) {
        buf[len] = '\0';
        return std::string(buf);
    }
    return std::string(); // Empty string if there's an error
}


static int readmesh(const char* filename, Array3D<scalar,_NF> &data, Array3D<int,2> &neighbours){
  std::string executablePath = getExecutablePath();
  std::string dataPath = executablePath.substr(0, executablePath.find_last_of("/")) + meshfile;
  FILE *fp = fopen(dataPath.c_str(),"r");
  if(!fp){
    printf("Could not open file %s\n",filename);
    exit(1);
  }
  int n_elem;
  fscanf(fp,"%d",&n_elem);
  int nelem_pad = AOSOA * ((n_elem+AOSOA-1)/AOSOA);
  data.data = malloc_manged_flags<scalar>(_NF*3*nelem_pad);
  data.n_elem = n_elem;
  neighbours.data = malloc_manged_flags<int>(2*3*nelem_pad);
  neighbours.n_elem = n_elem;

  for(int elem = 0; elem < n_elem; elem++){
    for(int node = 0; node <3; node++){
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

static void write_result(const char* filename, Array3D<scalar,3> &solution){
  FILE *fp = fopen(filename,"w");
  int n_elem = solution.n_elem;
  for(int elem = 0; elem < n_elem; elem++){
    for(int node = 0; node <3; node++){
      fprintf(fp,"%e,%e,%e\n", solution(0,node,elem), solution(1,node,elem), solution(2,node,elem));
    }
  }
  fclose(fp);
}

template<int n_fields>
__device__ LocalArray3D<n_fields> load_elem(const Array3D<scalar, n_fields> &array, int elem) {
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


template<int n_fields>
class SharedArray3D {
  public:
  v_scalar data[BLOCK_SIZE*n_fields*3];
  __device__ constexpr static int num_fields(){return n_fields;};
  __device__ inline v_scalar &operator()(int field, int node, int elem) {
    return data[BLOCK_SIZE*3*field + BLOCK_SIZE*node];
  }
};

template<typename Array>
__device__ inline static v_scalar fe_2d_interp_field(Array &solution, int field, v_index ie, const v_scalar phi[3]){
  v_scalar v = 0;
  for (int node = 0; node <3; ++node) {
    v += solution(field, node, ie)*phi[node];
  }
  return v;
}


template<typename Array>
__device__ inline static v_scalar fe_2d_interp_field(Array &solution, int field, int ie, const scalar phi[3]){
  v_scalar v = 0;
  for (int node = 0; node <3; ++node) {
    v += solution(field, node, ie)*phi[node];
  }
  return v;
}

template<typename Array>
__device__ void inline static fe_2d_triangle(Array &xy, int elem, v_scalar dphi[3][2], v_scalar &jac) {
  v_scalar dxdxi[2][2]={{0}};
  v_scalar dxidx[2][2];
  for (int node = 0; node <3; node++) {
    dxdxi[0][0] += xy(0, node, elem)*tri_dphi_dxi[node][0];
    dxdxi[0][1] += xy(0, node, elem)*tri_dphi_dxi[node][1];
    dxdxi[1][0] += xy(1, node, elem)*tri_dphi_dxi[node][0];
    dxdxi[1][1] += xy(1, node, elem)*tri_dphi_dxi[node][1];
  }
  jac = inv2x2(dxdxi,dxidx);
  for (int node = 0; node <3; node++) {
    for (int j = 0; j < 2; ++j) {
      dphi[node][j] = tri_dphi_dxi[node][0]*dxidx[0][j] + tri_dphi_dxi[node][1]*dxidx[1][j];
    }
  } 
}


template<typename Array>
__device__ void inline static fe_2d_triangle(Array &xy, v_index elem, v_scalar dphi[3][2], v_scalar &jac) {
  v_scalar dxdxi[2][2]={{0}};
  v_scalar dxidx[2][2];
  for (int node = 0; node <3; node++) {
    dxdxi[0][0] += xy(0, node, elem)*tri_dphi_dxi[node][0];
    dxdxi[0][1] += xy(0, node, elem)*tri_dphi_dxi[node][1];
    dxdxi[1][0] += xy(1, node, elem)*tri_dphi_dxi[node][0];
    dxdxi[1][1] += xy(1, node, elem)*tri_dphi_dxi[node][1];
  }
  jac = inv2x2(dxdxi,dxidx);
  for (int node = 0; node <3; node++) {
    for (int j = 0; j < 2; ++j) {
      dphi[node][j] = tri_dphi_dxi[node][0]*dxidx[0][j] + tri_dphi_dxi[node][1]*dxidx[1][j];
    }
  } 
}

template<typename Array>
__device__ inline static void fe_2d_assemble_term(Array &rhse,int nphi, int nf, scalar *phi, v_scalar dphi[3][2], v_scalar w, v_scalar *s) {
  for (int node = 0; node < nphi; node++) { 
    for (int field = 0; field < nf; ++field) {
      v_scalar r = phi[node]*s[field];
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

__device__ inline static void tri_phi_e(scalar pts[2], scalar phi[3]) {
  scalar xi = pts[0], eta = pts[1];
  phi[0] = 1-xi-eta;
  phi[1] = xi;
  phi[2] = eta;
}

__device__ inline static void tri_phi_e(v_scalar pts[2], v_scalar phi[3]) {
  v_scalar xi = pts[0], eta = pts[1];
  phi[0] = 1-xi-eta;
  phi[1] = xi;
  phi[2] = eta;
}

template<typename Array>
__device__ inline static void fe_2d_grad_field(Array &solution, int field, int ie, const v_scalar dphi[3][2], v_scalar v[2]){
  for (int k = 0; k < 2; ++k){
    v[k] = 0;
  }
  for (int node = 0; node <3; ++node) {
    for (int k = 0; k < 2; ++k){
      v[k] += solution(field, node, ie)*dphi[node][k];
    }
  }
}

#include <algorithm>
#include <execution>

template<int block_size=1, class K>
void ava_for(size_t from, size_t to, K k) {
  // if (block_size != 1) {
  //   printf("ava_for only supports block_size=1 in CPU, not %i\n", block_size);
  //   exit(1);
  // }
  std::for_each(std::execution::unseq, (char*)from, (char*)to, [k](auto &i) {k((size_t)&i);});
}


#endif

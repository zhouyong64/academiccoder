/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename t> t* d_allocateArray(int size, const char* file = 0, int linenumber = 0)
{
  //cout << file << " " << linenumber<<endl;
  t* d_ptr;
  if (cudaSuccess != cudaMalloc((void**) &d_ptr, sizeof(t)*size)) {
    cout<< "cudaMalloc error.";
    if (file != 0) cout << " Called from file: " << file << " linenumber: " << linenumber << endl;
    cout << endl;
    exit(1);
  }
  return d_ptr;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename t> void d_zeroArray(t* d_ptr, int count) {
  cudaMemset(d_ptr,  0,sizeof(t)*count);
}
template <typename t> t* d_allocateArrayZeroed(int size, const char* file = 0, int linenumber = 0)
{
  t* d_ptr = d_allocateArray<t>(size, file, linenumber);
  d_zeroArray<t>(d_ptr, size);
  return d_ptr;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
template <typename t> void h2dMemcopy(t* h_ptr, t* d_ptr, int size)
{
  cudaMemcpy(d_ptr, h_ptr, sizeof(t)*size, cudaMemcpyHostToDevice);
}
template <typename t> void d2hMemcopy(t* d_ptr, t* h_ptr, int size)
{
  cudaMemcpy(h_ptr, d_ptr, sizeof(t)*size, cudaMemcpyDeviceToHost);
}

template <typename t> t* d_allocateArrayFromVector(vector<t> &source, const char* file = 0, int linenumber = 0) {
  t* d_ptr = d_allocateArray<t>(source.size(), file, linenumber);
  h2dMemcopy<t>(&source[0],d_ptr,source.size());
  return d_ptr;
}

static void cublasError(cublasStatus_t error)
{
  switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
      break;

    case CUBLAS_STATUS_NOT_INITIALIZED:
      cout <<  "CUBLAS_STATUS_NOT_INITIALIZED\n";
      break;

    case CUBLAS_STATUS_ALLOC_FAILED:
      cout <<  "CUBLAS_STATUS_ALLOC_FAILED\n";
      break;

    case CUBLAS_STATUS_INVALID_VALUE:
      cout <<  "CUBLAS_STATUS_INVALID_VALUE\n";
      break;

    case CUBLAS_STATUS_ARCH_MISMATCH:
      cout <<  "CUBLAS_STATUS_ARCH_MISMATCH\n";
      break;

    case CUBLAS_STATUS_MAPPING_ERROR:
      cout <<  "CUBLAS_STATUS_MAPPING_ERROR\n";
      break;

    case CUBLAS_STATUS_EXECUTION_FAILED:
      cout <<  "CUBLAS_STATUS_EXECUTION_FAILED\n";
      break;

    case CUBLAS_STATUS_INTERNAL_ERROR:
      cout <<  "CUBLAS_STATUS_INTERNAL_ERROR\n";
      break;
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////
//GEMM for matrices in row major form. /////////////////////////////////////////////////////////// //////////////////////////////////////////////////////////////////////////////////////////////////
//A is l*m, B is m*r, C is l*r. Set C to alpha A B + beta C.
void d_rowMajorSGEMM_alphaAB_betaC (cublasHandle_t handle,
                                    float* A, float* B, float* C,
                                    int l, int m, int r,
                                    float alpha, float beta)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N,r,l,m,&alpha,B,r,A,m,&beta,C,r));
}
//A^t is l*m, B is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtB_betaC (cublasHandle_t handle,
                                     float* A, float* B, float* C,
                                     int l, int m, int r,
                                     float alpha, float beta)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_T,r,l,m,&alpha,B,r,A,l,&beta,C,r));
}
//A is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaABt_betaC (cublasHandle_t handle,
                                     float* A, float* B, float* C,
                                     int l, int m, int r,
                                     float alpha, float beta)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_N,r,l,m,&alpha,B,m,A,m,&beta,C,r));
}
//A^t is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtBt_betaC (cublasHandle_t handle,
                                      float* A, float* B, float* C,
                                      int l, int m, int r,
                                      float alpha, float beta)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_T,r,l,m,&alpha,B,m,A,l,&beta,C,r));
}
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename t> void printMatrix(vector<t> &m, int rows, int cols, int maxVal=10, string s="")
{
  cout << s << "-----------" <<endl;
  for (int r=0; r<rows && r<maxVal;r++) {
    for (int c=0;c<cols && c<maxVal;c++) {
      cout << m[r*cols+c] << "\t ";}
    cout <<"\n";}
  cout <<"--------------------------------------------------------------------------\n";
}
template <typename t> void d_printMatrix(t* d_ptr, int rows, int cols, int maxVal=10, string s="")
{
  vector<t> m(rows*cols);
  d2hMemcopy<t>(d_ptr,&m[0],rows*cols);
  printMatrix<t>(m,rows,cols,maxVal,s);
}

template <typename t> void printV(vector<t> v) {
  for(int i=0;i<v.size();i++)
    cout << v[i] << " ";
  cout <<endl;}
template <typename t> void printVV(vector<vector<t> > v) {
  cout << "<<"<<endl;
  for(int i=0;i<v.size();i++)
    printV(v[i]);
  cout <<">>"<<endl;}


float peek(float*d ) {
  float val;
  d2hMemcopy<float>(d,&val,1);
  return val;
}
void poke(float* d, float val) {
  h2dMemcopy<float>(&val,d,1);
}

void canary(const char* file = 0, int linenumber = 0) {
  for (int i=0;i<10;i++) {
    float* a=d_allocateArrayZeroed<float>(100,file,linenumber);
    cudaFree(a);
  }
}

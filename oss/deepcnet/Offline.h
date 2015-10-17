class Pixel{
public:
  short int x;
  short int y;
  vector<float> val;
};
typedef vector<Pixel> Pixels;

class OfflinePicture : public Picture {
public:
  Pixels pix;
  void codifyInputData (SparseCnnInterface &interface);
  Picture* distort ();
  ~OfflinePicture() {};
};

void OfflinePicture::codifyInputData (SparseCnnInterface &interface) {
  //Assume we need a null vector for the background
  for(int i=0;i<nInputFeatures;i++)
    interface.features.push_back(0);
  int backgroundNullVectorNumber=interface.count++;
  interface.backgroundNullVectorNumbers.push_back(backgroundNullVectorNumber);
  vector<int> grid(scale_N*scale_N,backgroundNullVectorNumber);

  for (int i=0; i<pix.size(); i++) {
    int n=(pix[i].x+scale_N/2)*scale_N+(pix[i].y+scale_N/2);
    grid[n]=interface.count++;
    for (int k=0;k<nInputFeatures;k++) {
      interface.features.push_back(pix[i].val[k]/regularizingConstants[k]);
    }
  }

  interface.grids.push_back(grid);
  while (interface.featureSampleNumbers.size() < interface.count)
    interface.featureSampleNumbers.push_back(interface.batchSize);
  interface.batchSize++;
  interface.labels.push_back(label);
}

void offlineJiggle(Pixels &pix, RNG &rng, int offlineJiggle)
{
  int dx=rng.randint(offlineJiggle*2+1)-offlineJiggle;
  int dy=rng.randint(offlineJiggle*2+1)-offlineJiggle;
  for (int j=0;j<pix.size();j++) {
    pix[j].x+=dx;
    pix[j].y+=dy;
  }
}

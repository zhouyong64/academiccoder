#include "FloatPoint.h"

class OfflineGridPicture : public Picture {
public:
  short int xOffset;
  short int yOffset;
  short int xSize;
  short int ySize;
  vector<float> bitmap; //nInputFeatures*ySize*xSize (row major order)
  void codifyInputData (SparseCnnInterface &interface);
  Picture* distort ();

  OfflineGridPicture(int xSize, int ySize, int label_ = -1) : xSize(xSize), ySize(ySize) {
    label=label_;
    xOffset=-xSize/2;
    yOffset=-ySize/2;
    bitmap.resize(nInputFeatures*ySize*xSize);
  }
  ~OfflineGridPicture() {}
  void jiggle(RNG &rng, int offlineJiggle) {
    xOffset+=rng.randint(offlineJiggle*2+1)-offlineJiggle;
    yOffset+=rng.randint(offlineJiggle*2+1)-offlineJiggle;
  }
  // void drawPGM(char* filename) {
  //   ofstream f(filename);
  //   f << "P2\n"<< xSize*nInputFeatures << " " << ySize<< endl<< 255<< endl;
  //   for (int y=0; y<ySize; y++) {
  //     for (int i=0; i<nInputFeatures;i++) {
  //       for (int x=0; x<xSize;x++) {
  //         f << (int)(127.9*bitmap[x+y*xSize+i*ySize*xSize]/regularizingConstants[i]+128) << " ";
  //       }
  //       f << endl;
  //     }
  //     f << endl;
  //   }
  //   f.close();
  // }
  void drawPGM(char* filename) {
    ofstream f(filename);
    f << "P2\n"<< scale_N*nInputFeatures << " " << scale_N<< endl<< 255<< endl;
    for (int y=-scale_N/2; y<scale_N/2; y++) {
      for (int i=0; i<nInputFeatures;i++) {
        for (int x=-scale_N/2; x<scale_N/2;x++) {
          FloatPoint p(x+0.5,y+0.5);
          f << (int)(127.9*interpolate(p,i)/regularizingConstants[i]+128) << " ";
        }
        f << endl;
      }
      f << endl;
    }
    f.close();
  }
  void drawPPM(char* filename) { //for when nInputFeatures==3
    ofstream f(filename);
    f << "P3\n"<< xSize<< " " << ySize<< endl<< 255<< endl;
    for (int y=0; y<ySize; y++) {
      for (int x=0; x<xSize;x++) {
        for (int col=0; col<3; col++) {
          f << 128+(int)bitmap[x+y*xSize+col*xSize*ySize] << " ";
        }
      }
      f << endl;
    }
    f << endl;
    f.close();
  }
  float interpolate(FloatPoint& p, int i) {
    return bilinearInterpolationScaled<float>
      (p, &bitmap[i*xSize*ySize],
       xSize, ySize,
       xOffset,       yOffset,
       xOffset+xSize, yOffset+ySize);
  }
};

void OfflineGridPicture::codifyInputData (SparseCnnInterface &interface) {
  for  (int i=0; i<nInputFeatures; i++)
    interface.features.push_back(0); //Background feature
  int backgroundNullVectorNumber=interface.count++;
  interface.backgroundNullVectorNumbers.push_back(backgroundNullVectorNumber);
  vector<int> grid(scale_N*scale_N,backgroundNullVectorNumber);
  for (int x=0; x<xSize; x++) {
    for (int y=0; y<ySize; y++) {
      if (x+xOffset+scale_N/2>=0 && x+xOffset+scale_N/2<scale_N &&
          y+yOffset+scale_N/2>=0 && y+yOffset+scale_N/2<scale_N) {
        bool flag=false;
        for (int i=0; i<nInputFeatures; i++)
          if (abs(bitmap[x+y*xSize+i*xSize*ySize])>0.005*regularizingConstants[i])
            flag=true;
        if (flag) {
          int n=(x+xOffset+scale_N/2)*scale_N+(y+yOffset+scale_N/2);
          grid[n]=interface.count++;
          for (int i=0; i<nInputFeatures; i++)
            interface.features.push_back
              (bitmap[x+y*xSize+i*xSize*ySize]/regularizingConstants[i]);
        }
      }
    }
  }
  interface.grids.push_back(grid);
  while (interface.featureSampleNumbers.size() < interface.count)
    interface.featureSampleNumbers.push_back(interface.batchSize);
  interface.batchSize++;
  interface.labels.push_back(label);
}

#include "readNorbSmall.h"
#include "readCIFAR10.h"
#include "readCIFAR100.h"
#include "readMNIST.h"
#include "readOfflineCasia.h"


//Example distortion functions

// Picture* OfflineGridPicture::distort() {
//   OfflineGridPicture* pic=new OfflineGridPicture(*this);
//   RNG rng;
//   pic->jiggle(rng,2);
//   // char filenameFormat[]="%d.pgm", filename[100];
//   // sprintf(filename,filenameFormat,rng.randint(1000));pic->drawPGM(filename);
//   return pic;
// }

// Picture* OfflineGridPicture::distort() {
//   OfflineGridPicture* pic=new OfflineGridPicture(xSize+40,ySize+40,label);
//   RNG rng;
//   float xStretch=rng.uniform(-0.1,0.1);
//   float yStretch=rng.uniform(-0.1,0.1);
//   //int flip_h=rng.randint(2);
//   int r=rng.randint(3);
//   float alpha=rng.uniform(-0.1,0.1);

//   for (int y=0; y<pic->ySize; y++)
//     for (int x=0; x<pic->xSize;x++) {
//       FloatPoint p(x+pic->xOffset+0.5,y+pic->yOffset+0.5);
//       p.stretch_x(xStretch);
//       p.stretch_y(yStretch);
//       //if (flip_h==1) p.flip_horizontal();
//       if (r==0) p.rotate(alpha);
//       if (r==1) p.slant_x(alpha);
//       if (r==2) p.slant_y(alpha);
//       for (int i=0; i<nInputFeatures; i++)
//         pic->bitmap[x+y*pic->xSize+i*pic->xSize*pic->ySize]=interpolate(p, i);
//     }
//   pic->jiggle(rng,2);
//   //char filenameFormat[]="%d.ppm", filename[100];
//   //sprintf(filename,filenameFormat,rng.randint(1000));pic->drawPGM(filename);
//   return pic;
// }

// Picture* OfflineGridPicture::distort() {
//   OfflineGridPicture* pic=new OfflineGridPicture(xSize+40,ySize+40,label);
//   EDfield edf(24,28,10,1);
//   for (int y=0; y<pic->ySize; y++)
//     for (int x=0; x<pic->xSize;x++) {
//       FloatPoint p(x+pic->xOffset+0.5,y+pic->yOffset+0.5);
//       p.stretch(edf);
//       p.stretch(edf);
//       p.stretch(edf);
//       for (int i=0; i<nInputFeatures; i++)
//         pic->bitmap[x+y*pic->xSize+i*pic->xSize*pic->ySize]=interpolate(p, i);
//     }
//   RNG rng;
//   pic->jiggle(rng,2);
//   // char filenameFormat[]="%d.pgm", filename[100];
//   // sprintf(filename,filenameFormat,rng.randint(1000));pic->drawPGM(filename);
//   return pic;
// }


// EDfields edf(pow(10,5),24,28,6,3);
// Picture* OfflineGridPicture::distort() {
//   OfflineGridPicture* pic=new OfflineGridPicture(xSize+40,ySize+40,label);
//   RNG rng;
//   int ind=rng.index(edf.edf);
//   for (int y=0; y<pic->ySize; y++)
//     for (int x=0; x<pic->xSize;x++) {
//       FloatPoint p(x+pic->xOffset+0.5,y+pic->yOffset+0.5);
//       p.stretch(edf.edf[ind]);
//       for (int i=0; i<nInputFeatures; i++)
//         pic->bitmap[x+y*pic->xSize+i*pic->xSize*pic->ySize]=interpolate(p, i);
//     }
//   return pic;
// }


// void OfflineGrid_test_show_errors() {
//   RNG rng;
//   int mistakes=0;
//   for (int ep=0; ep<testCharacters.size(); ep+=trainingBatchSize) {
//     SparseCnnInterface* batch = new SparseCnnInterface(TESTBATCH);
//     for (int i=ep;i<min<int>(ep+trainingBatchSize,testCharacters.size());i++) {
//       testCharacters[i]->codifyInputData(*batch);
//     }
//     ccnn.processBatch(batch);
//     for (int i=ep;i<min<int>(ep+trainingBatchSize,testCharacters.size());i++) {
//       if (batch->labels[i-ep]!=batch->topGuesses[i-ep][0]) {
//         char filenameFormat[]="%d_label%d_prediction_%d.pgm", filename[100];
//         sprintf(filename,filenameFormat,mistakes++,batch->labels[i-ep],batch->topGuesses[i-ep][0]);
//         dynamic_cast<OfflineGridPicture*>(testCharacters[i])->drawPGM(filename);
//       }
//     }
//     delete batch;
//   }
// }

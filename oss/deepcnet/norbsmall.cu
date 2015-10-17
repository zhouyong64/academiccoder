const int scale_N=96;
const float learningRate=0.001;
const int trainingBatchSize=100;
const int startEpoch=0*1000;
const int nInputFeatures=2;
const int nCharacters=10;
const int offlineMaxJiggle = 8;
const char weightFileNameFormat[]="norbsmall_epoch-%d.cnn";
#define ACTION train_test()
#include "CNN.h"
#define NORBSMALL
#include "OfflineRGB.h"

DeepCNet cnn(6,50,nInputFeatures,nCharacters,learningRate,0.1,0,startEpoch);

Picture* OfflineGridPicture::distort() {
  OfflineGridPicture* pic=new OfflineGridPicture(*this);
  RNG rng;
  pic->jiggle(rng,2);
  // char filenameFormat[]="%d.pgm", filename[100];
  // sprintf(filename,filenameFormat,rng.randint(1000));pic->drawPGM(filename);
  return pic;
}

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
//   char filenameFormat[]="%d.ppm", filename[100];
//   sprintf(filename,filenameFormat,rng.randint(1000));pic->drawPGM(filename);
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
#include "run.h"

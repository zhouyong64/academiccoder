const int scale_N=192;
const float learningRate=0.003;
const float learningRateDecayRate=pow(10,-5);
const int trainingBatchSize=100;
const int startEpoch=0*1000;
const int nInputFeatures=1;
const int nCharacters=3755;
const char weightFileNameFormat[]="casiaOffline_epoch-%d.cnn";
#define ACTION train_test()
#include "CNN.h"
#define CASIAOFFLINE
#include "OfflineGridUByte.h"
DeepCNet cnn(6,100,nInputFeatures,nCharacters,learningRate,0.1,0.0,startEpoch);


Picture* OfflineGridUBytePicture::distort() {
  OfflineGridUBytePicture* pic=new OfflineGridUBytePicture(xSize+40,ySize+40,label);
  RNG rng;
  float xStretch=rng.uniform(-0.1,0.1);
  float yStretch=rng.uniform(-0.1,0.1);
  int flip_h=rng.randint(2);
  int r=rng.randint(3);
  float alpha=rng.uniform(-0.1,0.1);

  for (int y=0; y<pic->ySize; y++)
    for (int x=0; x<pic->xSize;x++) {
      FloatPoint p(x+pic->xOffset+0.5,y+pic->yOffset+0.5);
      p.stretch_x(xStretch);
      p.stretch_y(yStretch);
      if (flip_h==1) p.flip_horizontal();
      if (r==0) p.rotate(alpha);
      if (r==1) p.slant_x(alpha);
      if (r==2) p.slant_y(alpha);
      for (int i=0; i<nInputFeatures; i++)
        pic->bitmap[x+y*pic->xSize+i*pic->xSize*pic->ySize]=interpolate(p, i);
    }
  pic->jiggle(rng,10);
  return pic;
}

#include "run.h"

const int scale_N=96;
const float learningRate=0.003;
const float learningRateDecayRate=pow(10,-5);
const int trainingBatchSize=100;
const int nInputFeatures=3;
const int startEpoch=0*1000;
const int nCharacters=100;
const char weightFileNameFormat[]="cifar100_epoch-%d.cnn";

#define ACTION train_test(2000,2000)
#include "CNN.h"
#define CIFAR100
#include "OfflineGrid.h"

FlatDeepCNet cnn(5,500,nInputFeatures,nCharacters,learningRate,0.1,0,startEpoch
                 ,list_of(0.0)(0.0)(0.1)(0.2)(0.3)(0.4)(0.5)
                 );

// Picture* OfflineGridPicture::distort() {
//   OfflineGridPicture* pic=new OfflineGridPicture(*this);
//   RNG rng;
//   pic->jiggle(rng,0);
//   return pic;
// }


Picture* OfflineGridPicture::distort() {
  OfflineGridPicture* pic=new OfflineGridPicture(xSize+40,ySize+40,label);
  RNG rng;
  float xStretch=rng.uniform(-0.2,0.2);
  float yStretch=rng.uniform(-0.2,0.2);
  int flip_h=rng.randint(2);
  int r=rng.randint(3);
  float alpha=rng.uniform(-0.2,0.2);

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
  pic->jiggle(rng,16);
  return pic;
}

#include "run.h"

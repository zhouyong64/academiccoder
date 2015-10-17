#include "gbcodes3755.h"
#pragma pack(push, 1)
struct gntCharacterHeader{
  unsigned int sampleSize;
  unsigned short label;
  unsigned short width;
  unsigned short height;
};
#pragma pack(pop)

#ifdef CASIAOFFLINE
void readGNTFile(vector<Picture*> &characters, const char* filename, bool numberLabelsFromZero = true) {
  ifstream file(filename,ios::in|ios::binary);
  if (!file) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);}
  cout << "\n" << filename << endl;
  gntCharacterHeader gCH;
  while (file.read((char*)&gCH,sizeof(gntCharacterHeader))) {
    OfflineGridUBytePicture* character = new OfflineGridUBytePicture(gCH.width,gCH.height,gCH.label);
    unsigned char *bitmap=new unsigned char[gCH.width*gCH.height];
    file.read((char *)bitmap,gCH.width*gCH.height);
    for (int i=0;i<gCH.width*gCH.height;i++){  //Black (0) background, white (255) ink
      character->bitmap[i]=255-bitmap[i];
      // if (i%gCH.width==0) cout << endl;
      // if (bitmap[i]==255)
      //   cout << " ";
      // else
      //   cout << "x";
    }
    if (numberLabelsFromZero) {
      character->label=find(gbcodesCRO,gbcodesCRO+3755,character->label)-gbcodesCRO;
      if (character->label<nCharacters)
        characters.push_back(character);
      else
        delete character;
    } else {
      characters.push_back(character);
    }
  }
  file.close();
}
void loadData()
{
  char filenameFormatA[]="Data/CASIA_gnt_files/%03d-f.gnt";
  char filenameFormatB[]="Data/CASIA_gnt_files/%04d-f.gnt";
  char filename[100];
  for(int fileNumber=1241;fileNumber<1300;fileNumber++) {
    sprintf(filename,filenameFormatB,fileNumber);
    readGNTFile(testCharacters,filename);
    cout <<" " << trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=1;fileNumber<=420;fileNumber++) {
    sprintf(filename,filenameFormatA,fileNumber);
    readGNTFile(trainCharacters,filename);
    cout <<" " << trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=501;fileNumber<=800;fileNumber++) {
    sprintf(filename,filenameFormatA,fileNumber);
    readGNTFile(trainCharacters,filename);
    cout <<" " << trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=1001;fileNumber<=1240;fileNumber++) {
    sprintf(filename,filenameFormatB,fileNumber);
    readGNTFile(trainCharacters,filename);
    cout <<" " << trainCharacters.size()<< " " << testCharacters.size();
  }
  cout << endl;
}
#endif

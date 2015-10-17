#ifdef CIFAR10
void readBINFile(vector<Picture*> &characters, const char* filename, bool mirror=false) {
  ifstream file(filename,ios::in|ios::binary);
  if (!file) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);
  }
  cout << "\r" << filename;
  unsigned char label;
  while (file.read((char*)&label,1)) {
    OfflineGridPicture* character = new OfflineGridPicture(32,32,label);
    unsigned char bitmap[3072];
    file.read((char*)bitmap,3072);
    for (int i=0;i<3072;i++)
      character->bitmap[i]=bitmap[i]-128; //Grey == (0,0,0)
    characters.push_back(character);
  }
  file.close();
}
void loadData()
{
  char filenameFormat[]="Data/CIFAR10/data_batch_%d.bin";
  char filename[100];
  for(int fileNumber=1;fileNumber<=5;fileNumber++) {
    sprintf(filename,filenameFormat,fileNumber);
    readBINFile(trainCharacters,filename,true);
    cout <<" " << trainCharacters.size()<< " " << testCharacters.size();
  }
  char filenameTest[]="Data/CIFAR10/test_batch.bin";
  readBINFile(testCharacters,filenameTest);
  cout <<" " << trainCharacters.size()<< " " << testCharacters.size() << endl;
}
#endif

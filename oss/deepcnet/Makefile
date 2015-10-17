deepcnet: CNN.h convolutional-layer.h device-memory-functions.h nn-computational.h FloatPoint.h RNG.h run.h Online.h OfflineGrid.h
	
mnist: deepcnet mnist.cu readMNIST.h
	nvcc -o DeepCNet mnist.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet

mnistSmall: deepcnet mnistSmall.cu readMNIST.h
	nvcc -o DeepCNet mnistSmall.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet

mnistMaxoutSmall: deepcnet mnistSmall.cu readMNIST.h
	nvcc -o DeepCNet mnistMaxout.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet

ujipenchars: deepcnet ujipenchars.cu readUJIpenchars.h
	nvcc -o DeepCNet ujipenchars.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet

assamese: deepcnet assamese.cu readAssamese.h
	nvcc -o DeepCNet assamese.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet

pendigits: deepcnet pendigits.cu readPendigits.h
	nvcc -o DeepCNet pendigits.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet

casiaOnline: deepcnet casiaOnline.cu readOnlineCasia.h
	nvcc -o DeepCNet casiaOnline.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet
	
casiaOffline: deepcnet casiaOffline.cu readOfflineCasia.h
	nvcc -o DeepCNet casiaOffline.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet

cifar10: deepcnet cifar10.cu readCIFAR10.h
	nvcc -o DeepCNet cifar10.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet

cifar100: deepcnet cifar100.cu readCIFAR100.h
	nvcc -o DeepCNet cifar100.cu -lrt -lcublas -lboost_thread -arch sm_30 -O2
	./DeepCNet

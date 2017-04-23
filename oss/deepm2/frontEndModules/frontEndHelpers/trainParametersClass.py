# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from deepmedic import samplingType

class TrainSessionParameters(object) :
    #THE LOGIC WHETHER I GOT A PARAMETER THAT I NEED SHOULD BE IN HERE!
    
    #To be called from outside too.
    @staticmethod
    def getDefaultSessionName() :
        return "trainSession"
    
    #REQUIRED:
    @staticmethod
    def errorRequireChannelsTraining():
        print "ERROR: Parameter \"channelsTraining\" needed but not provided in config file. \
        This parameter should provide paths to files, as many as the channels (modalities) of the task. \
        Each of the files should contain a list of paths, one for each case to train on. \
        These paths in a file should point to the .nii(.gz) files that are the corresponding channel for a patient. \
        Please provide it in the format: channelsTraining = [\"path-to-file-for-channel1\", ..., \
        \"path-to-file-for-channelN\"]. The paths should be given in quotes, separated by commas \
        (list of strings, python-style). Exiting."; exit(1) 
    errReqChansTr = errorRequireChannelsTraining
    @staticmethod
    def errorRequireGtLabelsTraining():
        print "ERROR: Parameter \"gtLabelsTraining\" needed but not provided in config file. This parameter should \
        provide the path to a file. That file should contain a list of paths, one for each case to train on. \
        These paths should point to the .nii(.gz) files that contain the corresponding Ground-Truth labels for a case. \
        Please provide it in the format: gtLabelsTraining = \"path-to-file\". The path should be given in quotes\
         (a string, python-style). Exiting."; exit(1) 
    errReqGtTr = errorRequireGtLabelsTraining
    
    
    @staticmethod
    def errorRequireSamplMasksAreProbabMapsTrain() :
        print "ERROR: Parameter \"samplingMasksAreProbabMapsTrain\" needed but not provided in config file. \
        This parameters is needed when parameter \"useDefaultTrainingSamplingFromGtAndRoi\" = False, \
        in order to know whether the provided masks are probability maps or segmentation labels. \
        Please provide parameter in the form: samplingMasksAreProbabMapsTrain = True/False. \
        True if the masks given at \"masksForPos(Neg)SamplingTrain\" are probability maps \
        (can be non-normalized, like weights), or False if they are binary segmentation masks. Exiting."; exit(1) 
    errReqMasksTypeTr = errorRequireSamplMasksAreProbabMapsTrain
    @staticmethod
    def warnDefaultPosSamplMasksTrain() :
        print "WARN: Parameter \"weightedMapsForPosSamplingTrain\" was not provided in config file, even though \
        advanced training options were triggered by setting \"useDefaultTrainingSamplingFromGtAndRoi\" = False. \
        This parameter should point to a file that lists the paths (per case/patient) to weighted-maps that indicate \
        where to perform the sampling of training samples. Can be provided in the format: \
        weightedMapsForPosSamplingTrain = \"path-to-file\". The target file should have one entry per case (patient).\
         Each of them should be pointing to .nii(.gz) files that indicate where to extract the positive samples. \
         \n\tDEFAULT: In the case it is not provided (like now!) the corresponding samples are extracted uniformly \
         from the whole volume!"
        return None 
    warnDefPosMasksTr = warnDefaultPosSamplMasksTrain
    @staticmethod
    def warnDefaultNegSamplMasksTrain() :
        print "WARN: Parameter \"weightedMapsForNegSamplingTrain\" was not provided in config file, even though\
         advanced training options were triggered by setting \"useDefaultTrainingSamplingFromGtAndRoi\" = False. \
         This parameter should point to a file that lists the paths (per case/patient) to weighted-maps that indicate \
         where to perform the sampling of training samples. Can be provided in the format: \
         weightedMapsForNegSamplingTrain = \"path-to-file\". The target file should have one entry per case (patient). \
         Each of them should be pointing to .nii(.gz) files that indicate where to extract the negative samples. \
         \n\tDEFAULT: In the case it is not provided (like now!) the corresponding samples are extracted uniformly \
         from the whole volume!"
        return None
    warnDefNegMasksTr = warnDefaultNegSamplMasksTrain
    @staticmethod
    def errorRequirePredefinedLrSched() :
        print "ERROR: Parameter \"stable0orAuto1orPredefined2orExponential3LrSchedule\" was set to 2, to use \
        Predefined Schedule for Learning-Rate decrease, but no Predefined Schedule was given. Please specify at which \
        epochs to lower the Learning Rate, by providing the corresponding parameter in the format: \
        predefinedSchedule = [epoch-for-1st-decrease, ..., epoch-for-last-decrease], where the epochs are specified by \
        an integer > 0. Exiting."; exit(1) 
    errReqPredLrSch = errorRequirePredefinedLrSched
    @staticmethod
    def errorAutoRequiresValSamples() :
        print "ERROR: Parameter \"stable0orAuto1orPredefined2orExponential3LrSchedule\" was set to 1 (AUTO schedule). \
        This schedule requires performing validation on samples throughout training, because this schedule lowers \
        the Learning Rate when validation-accuracy plateaus. However the parameter \
        \"performValidationOnSamplesThroughoutTraining\" was set to False in the configuration file, or was ommitted,\
         which triggers the default value, False! Please set the parameter \
         performValidationOnSamplesThroughoutTraining = True. You will then need to provide the path to the channels \
         of the validation cases in the format: channelsValidation = \
         [\"path-to-file-that-lists-paths-to-channel-1-for-every-case\", ..., \
         \"path-to-file-that-lists-paths-to-channel-N-for-every-case\"] (python style list-of-strings). Also, you will \
         need to provide the Ground-Truth for the validation cases, in the format:  gtLabelsValidation = \"path-to-file\",\
          where the file lists the paths to the GT labels of each validation case. Exiting!"; exit(1)
    @staticmethod
    def errorRequireChannelsVal() :
        print "ERROR: Parameter \"channelsValidation\" was not provided, although it is required to perform validation, \
        although validation was requested (parameters \"performValidationOnSamplesThroughoutTraining\" or \
        \"performFullInferenceOnValidationImagesEveryFewEpochs\" was set to True). You will need to provide a list with \
        path to files that list where the channels for each validation case can be found. The corresponding parameter \
        must be provided in the format: channelsValidation = [\"path-to-file-that-lists-paths-to-channel-1-for-every-case\",\
         ..., \"path-to-file-that-lists-paths-to-channel-N-for-every-case\"] (python style list-of-strings). Exiting."; 
        exit(1)
    errReqChannsVal = errorRequireChannelsVal
    @staticmethod
    def errorReqGtLabelsVal() :
        print "ERROR: Parameter \"gtLabelsValidation\" was not provided, although it is required to perform validation \
        on training-samples, which was requested (parameter \"performValidationOnSamplesThroughoutTraining\" \
        was set to True). It is also useful so that the DSC score is reported if full-inference on the validation \
        samples is performed (when parameter \"performFullInferenceOnValidationImagesEveryFewEpochs\" is set to True)!\
         You will need to provide the path to a file that lists where the GT labels for each validation case can be found.\
          The corresponding parameter must be provided in the format: gtLabelsValidation = \
          \"path-to-file-that-lists-GT-labels-for-every-case\" (python style string). Exiting."; exit(1)
        
        
    #VALIDATION
    @staticmethod
    def errorReqNumberOfEpochsBetweenFullValInfGreaterThan0() :
        print "ERROR: It was requested to perform full-inference on validation images by setting parameter \
        \"performFullInferenceOnValidationImagesEveryFewEpochs\" to True. For this, it is required to specify \
        the number of epochs between two full-inference procedures. This number was given equal to 0. Please specify\
         a number greater than 0, in the format: numberOfEpochsBetweenFullInferenceOnValImages = 1\
          (Any integer. Default is 1). Exiting!"; exit(1)
    @staticmethod
    def errorRequireNamesOfPredictionsVal() :
        print "ERROR: It was requested to perform full-inference on validation images by setting parameter \
        \"performFullInferenceOnValidationImagesEveryFewEpochs\" to True and then save some of the results \
        (segmentation maps, probability maps or feature maps), either manually or by default. For this, \
        it is required to specify the path to a file, which should contain names to give to the results. \
        Please specify the path to such a file in the format: namesForPredictionsPerCaseVal = \
        \"./validation/validationNamesOfPredictionsSimple.cfg\" (python-style string). Exiting!"; exit(1)
    @staticmethod
    def errorRequirePercentOfPosSamplesVal():
        print "ERROR: Advanced sampling was enabled by setting: useDefaultUniformValidationSampling = False. \
        This requires providing the percentage of validation samples that should be extracted as positives \
        (from the positive weight-map). Please specify a float between 0.0 and 1.0, eg in the format: \
        percentOfSamplesToExtractPositiveVal = 0.5. Exiting!"; exit(1)
    errReqPercPosTrVal = errorRequirePercentOfPosSamplesVal
    @staticmethod
    def warnDefaultPercentOfPosSamplesVal():
        print "WARN: Advanced sampling was enabled by setting: useDefaultUniformValidationSampling = False. \
        This requires providing the percentage of validation samples that should be extracted as positives \
        (from the positive weight-map). Please specify a float between 0.0 and 1.0, eg in the format: \
        percentOfSamplesToExtractPositiveVal = 0.5. \n\tDEFAULT: In the case not given (like now!) \
        default value of 0.5 is used!"
        return 0.5
    warnDefPercPosTrVal = warnDefaultPercentOfPosSamplesVal
    @staticmethod
    def warnDefaultPosSamplMasksVal() :
        print "WARN: Parameter \"weightedMapsForPosSamplingVal\" was not provided in config file, even though \
        advanced validation-sampling options were triggered by setting \"useDefaultUniformValidationSampling\" = False.\
         This parameter should point to a file that lists the paths (per case/patient) to weighted-maps that indicate \
         where to perform the sampling of validation samples. Can be provided in the format: \
         weightedMapsForPosSamplingVal = \"path-to-file\". The target file should have one entry per case (patient). \
         Each of them should be pointing to .nii(.gz) files that indicate where to extract the positive samples. \
         \n\tDEFAULT: In the case it is not provided (like now!) the corresponding samples are extracted uniformly \
         from the whole volume!"
        return None 
    warnDefPosMasksVal = warnDefaultPosSamplMasksVal
    @staticmethod
    def warnDefaultNegSamplMasksVal() :
        print "WARN: Parameter \"weightedMapsForNegSamplingVal\" was not provided in config file, \
        even though advanced  validation-sampling options were triggered by setting \
        \"useDefaultUniformValidationSampling\" = False. This parameter should point to a file that lists the \
        paths (per case/patient) to weighted-maps that indicate where to perform the sampling of validation samples.\
         Can be provided in the format: weightedMapsForNegSamplingVal = \"path-to-file\". The target file should have\
          one entry per case (patient). Each of them should be pointing to .nii(.gz) files that indicate where to \
          extract the negative samples. \n\tDEFAULT: In the case it is not provided (like now!) the corresponding \
          samples are extracted uniformly from the whole volume!"
        return None
    warnDefNegMasksVal = warnDefaultNegSamplMasksVal
    
    @staticmethod
    def errorRequireOptimizer012() :
        print "ERROR: The parameter \"sgd0orAdam1orRms2\" must be given 0,1 or 2. Omit for default. Exiting!"; exit(1)
    @staticmethod
    def errorRequireMomentumClass0Nestov1() :
        print "ERROR: The parameter \"classicMom0OrNesterov1\" must be given 0 or 1. Omit for default. Exiting!"; exit(1)
    @staticmethod
    def errorRequireMomValueBetween01() :
        print "ERROR: The parameter \"momentumValue\" must be given between 0.0 and 1.0 Omit for default. Exiting!"; exit(1)
    @staticmethod
    def errorRequireMomNonNorm0Norm1() :
        print "ERROR: The parameter \"momNonNorm0orNormalized1\" must be given 0 or 1. Omit for default. Exiting!"; exit(1)
        
    # Deprecated :
    @staticmethod
    def errorDeprecatedPercPosTraining():
        print "ERROR: Parameter \"percentOfSamplesToExtractPositiveTrain\" in the config file is now Deprecated! Please \
        remove this entry from the train-config file. If you do not want the default behaviour but instead wish to \
        specify the proportion of Foreground and Background samples yourself, please activate the \"Advanced Sampling\" \
        options (useDefaultTrainingSamplingFromGtAndRoi=False), choose type-of-sampling Foreground/Background \
        (typeOfSamplingForTraining = 0) and use the new variable \"proportionOfSamplesToExtractPerCategoryTraining\" \
        which replaces the functionality of the deprecated\
         (eg. proportionOfSamplesToExtractPerCategoryTraining = [0.3, 0.7]). Exiting."; exit(1) 
    errDeprPercPosTr = errorDeprecatedPercPosTraining
    
    def __init__(self,
                sessionName,
                sessionLogger,
                mainOutputAbsFolder,
                cnn3dInstance,
                cnnModelFilepath,
                
                #==================TRAINING====================
                folderForSessionCnnModels,
                listWithAListPerCaseWithFilepathPerChannelTrain,
                gtLabelsFilepathsTrain,
                
                #[Optionals]
                #~~~~~~~~~Sampling~~~~~~~
                roiMasksFilepathsTrain,
                # DEPRECATED! Kept for backwards compatibility and issuing an error/warning!
                percentOfSamplesToExtractPositTrain, 
                #~~~~~~~~~Advanced Sampling~~~~~~~
                useDefaultTrainingSamplingFromGtAndRoi,
                samplingTypeTraining, # ForeBackgr0_uniform1_fullImage2_PerClass3
                proportionOfSamplesPerCategoryTrain,
                #[ [weightmap1-case1, weightmap1-case2], ...[wmN-case1, wmN-case2] ]
                listOfAListPerWeightMapCategoryWithFilepathsOfAllCasesTrain, 
                
                #~~~~~~~~Training Cycle ~~~~~~~
                numberOfEpochs,
                numberOfSubepochs,
                numOfCasesLoadedPerSubepoch,
                segmentsLoadedOnGpuPerSubepochTrain,
                
                #~~~~~~~ Learning Rate Schedule ~~~~~~~
                #Auto requires performValidationOnSamplesThroughoutTraining and providedGtForValidationBool
                stable0orAuto1orPredefined2orExponential3LrSchedule,
                
                #Stable + Auto + Predefined.
                whenDecreasingDivideLrBy,
                #Stable + Auto
                numEpochsToWaitBeforeLoweringLr,
                #Auto:
                minIncreaseInValidationAccuracyThatResetsWaiting,
                #Predefined.
                predefinedSchedule,
                #Exponential
                exponentialSchedForLrAndMom,
                
                #~~~~~~~ Augmentation~~~~~~~~~~~~
                reflectImagesPerAxis,
                performIntAugm,
                sampleIntAugmShiftWithMuAndStd,
                sampleIntAugmMultiWithMuAndStd,
                
                #==================VALIDATION=====================
                performValidationOnSamplesThroughoutTraining,
                performFullInferenceOnValidationImagesEveryFewEpochs,
                #Required:
                listWithAListPerCaseWithFilepathPerChannelVal,
                gtLabelsFilepathsVal,
                
                segmentsLoadedOnGpuPerSubepochVal,
                
                #[Optionals]
                roiMasksFilepathsVal, #For default sampling and for fast inference. Optional. Otherwise from whole image.
                
                #~~~~~~~~Full Inference~~~~~~~~
                numberOfEpochsBetweenFullInferenceOnValImages,
                #Output                                
                namesToSavePredictionsAndFeaturesVal,
                #predictions
                saveSegmentationVal,
                saveProbMapsBoolPerClassVal,
                folderForPredictionsVal,
                #features:
                saveIndividualFmImagesVal,
                saveMultidimImgWithAllFmsVal,
                indicesOfFmsToVisualisePerPathwayAndLayerVal,
                folderForFeaturesVal,
                
                #~~~~~~~~ Advanced Validation Sampling ~~~~~~~~~~
                useDefaultUniformValidationSampling,
                samplingTypeValidation, # ForeBackgr0_uniform1_fullImage2_PerClass3
                proportionOfSamplesPerCategoryVal,
                #[ [weightmap1-case1, weightmap1-case2], ...[wmN-case1, wmN-case2] ]
                listOfAListPerWeightMapCategoryWithFilepathsOfAllCasesVal, 
                
                #====Optimization=====
                learningRate,
                optimizerSgd0Adam1Rms2,
                classicMom0Nesterov1,
                momentumValue,
                momNonNormalized0Normalized1,
                #Adam
                b1Adam,
                b2Adam,
                eAdam,
                #Rms
                rhoRms,
                eRms,
                #Regularization
                l1Reg,
                l2Reg,
                
                #~~~~ Freeze Layers ~~~~
                layersToFreezePerPathwayType,
                
                #==============Generic and Preprocessing===============
                padInputImagesBool
                ):
        
        #Importants for running session.
        
        self.sessionName = sessionName if sessionName else self.getDefaultSessionName()
        self.sessionLogger = sessionLogger
        self.mainOutputAbsFolder = mainOutputAbsFolder
        self.cnn3dInstance = cnn3dInstance #Must be filled from outside after initialization
        self.cnnModelFilepath = cnnModelFilepath #This is where the model was loaded from.
        self.cnnModelName = cnn3dInstance.cnnModelName if cnn3dInstance.cnnModelName <> None else "defaultCnnModelName"
        
        #====================TRAINING==========================
        self.pathAndFilenameToSaveTrainedModels = folderForSessionCnnModels + "/" + \
            self.cnnModelName + "." + self.sessionName
        #[[case1-ch1, case1-ch2], ..., [caseN-ch1, caseN-ch2]]
        self.channelsFilepathsTrain = listWithAListPerCaseWithFilepathPerChannelTrain if \
            listWithAListPerCaseWithFilepathPerChannelTrain else self.errReqChansTr()
        self.gtLabelsFilepathsTrain = gtLabelsFilepathsTrain if gtLabelsFilepathsTrain else self.errReqGtTr()
        
        #[Optionals]
        #~~~~~~~~~Sampling~~~~~~~
        self.providedRoiMasksTrain = True if roiMasksFilepathsTrain else False
        #For Int-Augm and for Mask-Where-Neg if no advanced-sampling-train.
        self.roiMasksFilepathsTrain = roiMasksFilepathsTrain if roiMasksFilepathsTrain else [] 
        
        if percentOfSamplesToExtractPositTrain <> None : #Deprecated. Issue error and ask for correction.
            self.errDeprPercPosTr()
        #~~~~~~~~~Advanced Sampling~~~~~~~
        #ADVANCED CONFIG IS DISABLED HERE IF useDefaultSamplingFromGtAndRoi = True!
        self.useDefaultTrainingSamplingFromGtAndRoi = useDefaultTrainingSamplingFromGtAndRoi if \
            useDefaultTrainingSamplingFromGtAndRoi <> None else True
        DEFAULT_SAMPLING_TYPE_TR = 0
        if self.useDefaultTrainingSamplingFromGtAndRoi :
            self.samplingTypeInstanceTrain = samplingType.SamplingType( self.sessionLogger, DEFAULT_SAMPLING_TYPE_TR, \
                                                                        self.cnn3dInstance.numberOfOutputClasses )
            numberOfCategoriesOfSamplesTr = self.samplingTypeInstanceTrain.getNumberOfCategoriesToSample()
            self.samplingTypeInstanceTrain.setPercentOfSamplesPerCategoryToSample( \
                                                    [1.0/numberOfCategoriesOfSamplesTr]*numberOfCategoriesOfSamplesTr )
            self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatTraining = None
        else :
            samplingTypeToUseTr = samplingTypeTraining if samplingTypeTraining <> None else DEFAULT_SAMPLING_TYPE_TR
            self.samplingTypeInstanceTrain = samplingType.SamplingType( self.sessionLogger, samplingTypeToUseTr, \
                                                                        self.cnn3dInstance.numberOfOutputClasses)
            if samplingTypeToUseTr in [0,3] and proportionOfSamplesPerCategoryTrain :
                self.samplingTypeInstanceTrain.setPercentOfSamplesPerCategoryToSample( proportionOfSamplesPerCategoryTrain )
            else :
                numberOfCategoriesOfSamplesTr = self.samplingTypeInstanceTrain.getNumberOfCategoriesToSample()
                self.samplingTypeInstanceTrain.setPercentOfSamplesPerCategoryToSample( \
                                                [1.0/numberOfCategoriesOfSamplesTr]*numberOfCategoriesOfSamplesTr )
                
            self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatTraining = \
                listOfAListPerWeightMapCategoryWithFilepathsOfAllCasesTrain #If None, following bool will turn False.
            
        self.providedWeightMapsToSampleForEachCategoryTraining = True if \
                self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatTraining else False 
        
        #~~~~~~~~ Training Cycle ~~~~~~~~~~~
        self.numberOfEpochs = numberOfEpochs if numberOfEpochs <> None else 35
        self.numberOfSubepochs = numberOfSubepochs if numberOfSubepochs <> None else 20
        self.numOfCasesLoadedPerSubepoch = numOfCasesLoadedPerSubepoch if numOfCasesLoadedPerSubepoch <> None else 50
        self.segmentsLoadedOnGpuPerSubepochTrain = segmentsLoadedOnGpuPerSubepochTrain if \
            segmentsLoadedOnGpuPerSubepochTrain <> None else 1000
        
        #~~~~~~~ Learning Rate Schedule ~~~~~~~~
        #Auto requires performValidationOnSamplesThroughoutTraining and providedGtForValidationBool
        self.stable0orAuto1orPredefined2orExponential3LrSchedule = stable0orAuto1orPredefined2orExponential3LrSchedule if\
             stable0orAuto1orPredefined2orExponential3LrSchedule <> None else 3
        #Used for Stable + Auto + Predefined.
        self.whenDecreasingDivideLrBy = whenDecreasingDivideLrBy if whenDecreasingDivideLrBy <> None else 2.0
        #Stable + Auto. Set this to 0 to never lower it!
        self.numEpochsToWaitBeforeLoweringLr = numEpochsToWaitBeforeLoweringLr if \
            numEpochsToWaitBeforeLoweringLr <> None else 3
        #Auto:
        self.minIncreaseInValidationAccuracyThatResetsWaiting = minIncreaseInValidationAccuracyThatResetsWaiting if \
            minIncreaseInValidationAccuracyThatResetsWaiting <> None else 0.0005
        #Predefined.
        self.predefinedSchedLowerLrAtEpochs = predefinedSchedule
        if self.stable0orAuto1orPredefined2orExponential3LrSchedule == 2 and self.predefinedSchedLowerLrAtEpochs == None :
            self.errReqPredLrSch()
        #Exponential
        self.exponentialSchedForLrAndMom = exponentialSchedForLrAndMom if exponentialSchedForLrAndMom else\
             [self.numberOfEpochs/2, 1.0/(2**(8)),  0.9]
        
        #~~~~~~~~~~~~~~ Augmentation~~~~~~~~~~~~~~
        self.reflectImagesPerAxis = reflectImagesPerAxis if reflectImagesPerAxis else [False, False, False]
        self.performIntAugm = performIntAugm if performIntAugm <> None else False
        if self.performIntAugm :
            self.sampleIntAugmShiftWithMuAndStd = sampleIntAugmShiftWithMuAndStd if \
                sampleIntAugmShiftWithMuAndStd else [0.0 , 0.1]
            self.sampleIntAugmMultiWithMuAndStd = sampleIntAugmMultiWithMuAndStd if \
                sampleIntAugmMultiWithMuAndStd else [1.0 , 0.0]
            #Joe: this is the value passed to 'normAugmFlag' in do_training
            self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd = [2, 1,
                                        self.sampleIntAugmShiftWithMuAndStd,self.sampleIntAugmMultiWithMuAndStd]
        else :
            self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd = [0, "plcholder", [], []]
            
        #===================VALIDATION========================
        self.performValidationOnSamplesThroughoutTraining = performValidationOnSamplesThroughoutTraining if \
            performValidationOnSamplesThroughoutTraining <> None else False
        if self.stable0orAuto1orPredefined2orExponential3LrSchedule == 1 and \
                not self.performValidationOnSamplesThroughoutTraining :
            self.errorAutoRequiresValSamples()
        self.performFullInferenceOnValidationImagesEveryFewEpochs = performFullInferenceOnValidationImagesEveryFewEpochs if\
             performFullInferenceOnValidationImagesEveryFewEpochs <> None else False
        
        #Input:
        if self.performValidationOnSamplesThroughoutTraining or self.performFullInferenceOnValidationImagesEveryFewEpochs :
            #[[case1-ch1, case1-ch2], ..., [caseN-ch1, caseN-ch2]]
            self.channelsFilepathsVal = listWithAListPerCaseWithFilepathPerChannelVal if \
                listWithAListPerCaseWithFilepathPerChannelVal else self.errReqChannsVal()
        else :
            self.channelsFilepathsVal = []
        if self.performValidationOnSamplesThroughoutTraining :
            self.gtLabelsFilepathsVal = gtLabelsFilepathsVal if gtLabelsFilepathsVal <> None else self.errorReqGtLabelsVal()
        elif self.performFullInferenceOnValidationImagesEveryFewEpochs :
            self.gtLabelsFilepathsVal = gtLabelsFilepathsVal if gtLabelsFilepathsVal <> None else []
        else : # Dont perform either of the two validations.
            self.gtLabelsFilepathsVal = []
        self.providedGtVal = True if self.gtLabelsFilepathsVal <> None else False
        
        #[Optionals]
        self.providedRoiMasksVal = True if roiMasksFilepathsVal <> None else False #For fast inf.
        #Also for default sampling of neg segs.
        self.roiMasksFilepathsVal = roiMasksFilepathsVal if roiMasksFilepathsVal <> None else [] 
        
        #~~~~~Validation on Samples~~~~~~~~
        self.segmentsLoadedOnGpuPerSubepochVal = segmentsLoadedOnGpuPerSubepochVal if \
            segmentsLoadedOnGpuPerSubepochVal <> None else 3000
        
        #~~~~~~~~~Advanced Validation Sampling~~~~~~~~~~~
        #ADVANCED OPTION ARE DISABLED IF useDefaultUniformValidationSampling = True!
        self.useDefaultUniformValidationSampling = useDefaultUniformValidationSampling if \
            useDefaultUniformValidationSampling <> None else True
        DEFAULT_SAMPLING_TYPE_VAL = 1
        if self.useDefaultUniformValidationSampling :
            self.samplingTypeInstanceVal = samplingType.SamplingType( self.sessionLogger, DEFAULT_SAMPLING_TYPE_VAL, \
                                                                      self.cnn3dInstance.numberOfOutputClasses )
            numberOfCategoriesOfSamplesVal = self.samplingTypeInstanceVal.getNumberOfCategoriesToSample()
            self.samplingTypeInstanceVal.setPercentOfSamplesPerCategoryToSample( \
                                                [1.0/numberOfCategoriesOfSamplesVal]*numberOfCategoriesOfSamplesVal )
            self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatValidation = None
        else :
            samplingTypeToUseVal = samplingTypeValidation if samplingTypeValidation <> None else DEFAULT_SAMPLING_TYPE_VAL
            self.samplingTypeInstanceVal = samplingType.SamplingType( self.sessionLogger, \
                                                        samplingTypeToUseVal, self.cnn3dInstance.numberOfOutputClasses)
            if samplingTypeToUseVal in [0,3] and proportionOfSamplesPerCategoryVal :
                self.samplingTypeInstanceVal.setPercentOfSamplesPerCategoryToSample( proportionOfSamplesPerCategoryVal )
            else :
                numberOfCategoriesOfSamplesVal = self.samplingTypeInstanceVal.getNumberOfCategoriesToSample()
                self.samplingTypeInstanceVal.setPercentOfSamplesPerCategoryToSample( \
                                                [1.0/numberOfCategoriesOfSamplesVal]*numberOfCategoriesOfSamplesVal )
                
            self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatValidation = \
                listOfAListPerWeightMapCategoryWithFilepathsOfAllCasesVal #If None, following bool will turn False.
            
        self.providedWeightMapsToSampleForEachCategoryValidation = True if \
            self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatValidation else False 
        
        #~~~~~~Full inference on validation image~~~~~~
        self.numberOfEpochsBetweenFullInferenceOnValImages = numberOfEpochsBetweenFullInferenceOnValImages if \
            numberOfEpochsBetweenFullInferenceOnValImages <> None else 1
        if self.numberOfEpochsBetweenFullInferenceOnValImages == 0 and \
                self.performFullInferenceOnValidationImagesEveryFewEpochs :
            self.errorReqNumberOfEpochsBetweenFullValInfGreaterThan0()
            
        #predictions
        self.saveSegmentationVal = saveSegmentationVal if saveSegmentationVal <> None else True
        self.saveProbMapsBoolPerClassVal = saveProbMapsBoolPerClassVal if \
            (saveProbMapsBoolPerClassVal <> None and saveProbMapsBoolPerClassVal <> []) else \
                [True]*cnn3dInstance.numberOfOutputClasses
        #Filled by call to self.makeFilepathsForPredictionsAndFeatures()
        self.filepathsToSavePredictionsForEachPatVal = None 
        #features:
        self.saveIndividualFmImagesVal = saveIndividualFmImagesVal if saveIndividualFmImagesVal <> None else False
        self.saveMultidimImgWithAllFmsVal = saveMultidimImgWithAllFmsVal if \
            saveMultidimImgWithAllFmsVal <> None else False
        self.indicesOfFmsToVisualisePerPathwayAndLayerVal = [item if item <> None else [] for item in \
                            indicesOfFmsToVisualisePerPathwayAndLayerVal] #By default, save none.
        self.filepathsToSaveFeaturesForEachPatVal = None #Filled by call to self.makeFilepathsForPredictionsAndFeatures()
        
        #Output:
        #Given by the config file, and is then used to fill filepathsToSavePredictionsForEachPat and 
        # filepathsToSaveFeaturesForEachPat.
        self.namesToSavePredictionsAndFeaturesVal = namesToSavePredictionsAndFeaturesVal  
        if not self.namesToSavePredictionsAndFeaturesVal and \
                self.performFullInferenceOnValidationImagesEveryFewEpochs and\
                (self.saveSegmentationVal or True in self.saveProbMapsBoolPerClassVal or self.saveIndividualFmImagesVal \
                 or self.saveMultidimImgWithAllFmsVal) :
            self.errorRequireNamesOfPredictionsVal()
            
        #===================== OTHERS======================
        #Preprocessing
        self.padInputImagesBool = padInputImagesBool if padInputImagesBool <> None else True
        
        #Others useful internally or for reporting:
        self.numberOfCasesTrain = len(self.channelsFilepathsTrain)
        self.numberOfCasesVal = len(self.channelsFilepathsVal)
        self.numberOfClasses = cnn3dInstance.numberOfOutputClasses
        self.numberOfChannels = cnn3dInstance.numberOfImageChannelsPath1
        
        #HIDDENS, no config allowed for these at the moment:
        self.useSameSubChannelsAsSingleScale = True
        #List of Lists with filepaths per patient. Only used when above is False.
        self.subsampledChannelsFilepathsTrain = "placeholder" 
        #List of Lists with filepaths per patient. Only used when above is False.
        self.subsampledChannelsFilepathsVal = "placeholder" 
        # One element for Normal and one for Subsampled Pathway. Either None for no smoothing, 
        # or a List [std-r, std-c, std-z] of the gaussian kernel to smooth with.
        self.smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage = [None, None] 
        
        self.numberOfEpochsToWeightTheClassesInTheCostFunction = 0
        
        self._makeFilepathsForPredictionsAndFeaturesVal( folderForPredictionsVal, folderForFeaturesVal )
        
        #====Optimization=====
        self.learningRate = learningRate if learningRate <> None else 0.001
        self.optimizerSgd0Adam1Rms2 = optimizerSgd0Adam1Rms2 if optimizerSgd0Adam1Rms2 <> None else 2
        if self.optimizerSgd0Adam1Rms2 == 0 :
            self.b1Adam = "placeholder"; self.b2Adam = "placeholder"; self.eAdam = "placeholder";
            self.rhoRms = "placeholder"; self.eRms = "placeholder";
        elif self.optimizerSgd0Adam1Rms2 == 1 :
            self.b1Adam = b1Adam if b1Adam <> None else 0.9 #default in paper and seems good
            self.b2Adam = b2Adam if b2Adam <> None else 0.999 #default in paper and seems good
            self.eAdam = eAdam if eAdam else 10**(-8)
            self.rhoRms = "placeholder"; self.eRms = "placeholder";
        elif self.optimizerSgd0Adam1Rms2 == 2 :
            self.b1Adam = "placeholder"; self.b2Adam = "placeholder"; self.eAdam = "placeholder";
            self.rhoRms = rhoRms if rhoRms <> None else 0.9 #default in paper and seems good
            # 1e-6 was the default in the paper, but blew up the gradients in first try. Never tried 1e-5 yet.
            self.eRms = eRms if eRms <> None else 10**(-4) 
        else :
            self.errorRequireOptimizer012()
            
        self.classicMom0Nesterov1 = classicMom0Nesterov1 if classicMom0Nesterov1 <> None else 1
        if self.classicMom0Nesterov1 not in [0,1]:
            self.errorRequireMomentumClass0Nestov1()
        self.momNonNormalized0Normalized1 = momNonNormalized0Normalized1 if momNonNormalized0Normalized1 <> None else 1
        if self.momNonNormalized0Normalized1 not in [0,1] :
            self.errorRequireMomNonNorm0Norm1()
        self.momentumValue = momentumValue if momentumValue <> None else 0.6
        if self.momentumValue < 0. or self.momentumValue > 1:
            self.errorRequireMomValueBetween01()
            
        #==Regularization==
        self.l1Reg = l1Reg if l1Reg <> None else 0.000001
        self.l2Reg = l2Reg if l2Reg <> None else 0.0001
        
        #============= HIDDENS ==============
        # Indices of layers that should not be trained (kept fixed).
        indicesOfLayersToFreezeNorm = [ l-1 for l in layersToFreezePerPathwayType[0] ] if \
            layersToFreezePerPathwayType[0] <> None else []
        indicesOfLayersToFreezeSubs = [ l-1 for l in layersToFreezePerPathwayType[1] ] if \
            layersToFreezePerPathwayType[1] <> None else indicesOfLayersToFreezeNorm
        indicesOfLayersToFreezeFc = [ l-1 for l in layersToFreezePerPathwayType[2] ] if \
            layersToFreezePerPathwayType[2] <> None else []
        # Three sublists, one per pathway type: Normal, Subsampled, FC. eg: [[0,1,2],[0,1,2],[]
        self.indicesOfLayersPerPathwayTypeToFreeze = [ indicesOfLayersToFreezeNorm, indicesOfLayersToFreezeSubs, \
                                                        indicesOfLayersToFreezeFc ]
        
        self.costFunctionLetter = "L"
        
        """
        #NOTES: variables that have to do with number of pathways: 
                self.indicesOfLayersPerPathwayTypeToFreeze (="all" always currently. Hardcoded)
                self.useSameSubChannelsAsSingleScale, (always True currently)
                self.subsampledChannelsFilepathsTrain,
                self.subsampledChannelsFilepathsVal, (Deprecated. But I should support it in future, \
                cause it works well for non-scale pathways)
                indicesOfFmsToVisualisePerPathwayAndLayerVal (Repeat subsampled!)
        """
    def _makeFilepathsForPredictionsAndFeaturesVal(self,
                                            absPathToFolderForPredictionsFromSession,
                                            absPathToFolderForFeaturesFromSession
                                            ) :
        self.filepathsToSavePredictionsForEachPatVal = []
        self.filepathsToSaveFeaturesForEachPatVal = []
        if self.namesToSavePredictionsAndFeaturesVal <> None :
            for case_i in xrange(self.numberOfCasesVal) :
                filepathForCasePrediction = absPathToFolderForPredictionsFromSession + "/" + \
                    self.namesToSavePredictionsAndFeaturesVal[case_i]
                self.filepathsToSavePredictionsForEachPatVal.append( filepathForCasePrediction )
                filepathForCaseFeatures = absPathToFolderForFeaturesFromSession + "/" + \
                    self.namesToSavePredictionsAndFeaturesVal[case_i]
                self.filepathsToSaveFeaturesForEachPatVal.append( filepathForCaseFeatures )
                
    def printParametersOfThisSession(self) :
        logPrint = self.sessionLogger.print3
        logPrint("=============================================================")
        logPrint("=============== PARAMETERS FOR THIS SESSION =================")
        logPrint("=============================================================")
        logPrint("Session's name = " + str(self.sessionName))
        logPrint("CNN model's name = " + str(self.cnnModelName))
        logPrint("CNN model was loaded from = " + str(self.cnnModelFilepath))
        logPrint("~~Output~~")
        logPrint("Main output folder = " + str(self.mainOutputAbsFolder))
        logPrint("Path and filename to save trained models = " + str(self.pathAndFilenameToSaveTrainedModels))
        
        logPrint("~~~~~~~~~~~~~~~~~~Generic Information~~~~~~~~~~~~~~~~")
        logPrint("Number of Classes (incl. background) (from cnnModel) = " + str(self.numberOfClasses))
        logPrint("Number of Channels (Normal Pathway) (from cnnModel) = " + str(self.numberOfChannels))
        logPrint("Number of Cases for Training = " + str(self.numberOfCasesTrain))
        logPrint("Number of Cases for Validation = " + str(self.numberOfCasesVal))
        
        logPrint("~~~~~~~~~~~~~~~~~~Training parameters~~~~~~~~~~~~~~~~")
        logPrint("Filepaths to Channels of the Training Cases = " + str(self.channelsFilepathsTrain))
        logPrint("Filepaths to Ground-Truth labels of the Training Cases = " + str(self.gtLabelsFilepathsTrain))
        
        logPrint("~~Sampling~~")
        logPrint("Region-Of-Interest Masks provided = " + str(self.providedRoiMasksTrain))
        logPrint("Filepaths to ROI Masks of the Training Cases = " + str(self.roiMasksFilepathsTrain))
        
        logPrint("~~Advanced Sampling~~")
        logPrint("Using default sampling = " + str(self.useDefaultTrainingSamplingFromGtAndRoi) + \
                 ". NOTE: Adv.Sampl.Params are auto-set to perform default sampling if True.")
        logPrint("Type of Sampling = " + str(self.samplingTypeInstanceTrain.getStringOfSamplingType()) + \
                 " ("+ str(self.samplingTypeInstanceTrain.getIntSamplingType()) + ")")
        logPrint("Sampling Categories = " + str(self.samplingTypeInstanceTrain.getStringsPerCategoryToSample()) )
        logPrint("Percent of Samples to extract per Sampling Category = " + \
                 str(self.samplingTypeInstanceTrain.getPercentOfSamplesPerCategoryToSample()))
        logPrint("Provided Weight-Maps, pointing where to focus sampling for each category \
        (if False, samples will be extracted based on GT and ROI) = " + \
        str(self.providedWeightMapsToSampleForEachCategoryTraining))
        logPrint("Paths to weight-maps for sampling of each category = " + \
                 str(self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatTraining))
        
        logPrint("~~Training Cycle~~")
        logPrint("Number of Epochs = " + str(self.numberOfEpochs))
        logPrint("Number of Subepochs per epoch = " + str(self.numberOfSubepochs))
        logPrint("Number of cases to load per Subepoch (for extracting the samples for this subepoch) = " + \
                 str(self.numOfCasesLoadedPerSubepoch))
        logPrint("Number of Segments loaded on GPU per subepoch for Training = " + \
                 str(self.segmentsLoadedOnGpuPerSubepochTrain) + \
                 ". NOTE: This number of segments divided by the batch-size defines the number of \
                 optimization-iterations that will be performed every subepoch!")
        
        logPrint("~~Learning Rate Schedule~~")
        logPrint("Schedule Type: Stable (0), Auto (1), Predefined (2), Exponential (3) = " + \
                 str(self.stable0orAuto1orPredefined2orExponential3LrSchedule))
        logPrint("[Stable/Auto/Predefined] When decreasing Learning Rate, divide LR by = " + \
                 str(self.whenDecreasingDivideLrBy))
        logPrint("[Stable/Auto] Wait that many epochs before lowering LR = " + \
                 str(self.numEpochsToWaitBeforeLoweringLr))
        logPrint("[Auto] Minimum increase in validation accuracy (0. to 1.) that resets the waiting counter = " + \
                 str(self.minIncreaseInValidationAccuracyThatResetsWaiting))
        logPrint("[Predefined] Predefined Schedule of Epochs when the LR will be lowered = " + \
                 str(self.predefinedSchedLowerLrAtEpochs))
        logPrint("[Exponential] [Number of initial epochs to not change LR and Mom, \
        final value of LR, final value of Momentum] = " + str(self.exponentialSchedForLrAndMom))
        
        logPrint("~~Data Augmentation During Training~~")
        logPrint("Reflect images per axis = " + str(self.reflectImagesPerAxis))
        logPrint("Perform intensity-augmentation [I'= (I+shift)*mult] = " + str(self.performIntAugm))
        logPrint("[Int. Augm.] Sample Shift from N(mu,std) = " + \
                 str(self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd[2]))
        logPrint("[Int. Augm.] Sample Multi from N(mu,std) = " + \
                 str(self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd[3]))
        logPrint("[Int. Augm.] (DEBUGGING:) full parameters [no(0)/images(1)/segms(2), nonNorm(0)/norm(1), shift, mult] = " + \
                 str(self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd))
        
        logPrint("~~~~~~~~~~~~~~~~~~Validation parameters~~~~~~~~~~~~~~~~")
        logPrint("Perform Validation on Samples throughout training? = " + \
                 str(self.performValidationOnSamplesThroughoutTraining))
        logPrint("Perform Full Inference on validation cases every few epochs? = " + \
                 str(self.performFullInferenceOnValidationImagesEveryFewEpochs))
        logPrint("Filepaths to Channels of the Validation Cases (Req for either of the above) = " + \
                 str(self.channelsFilepathsVal))
        logPrint("Provided Ground-Truth for Validation = " + str(self.providedGtVal) + \
                 ". NOTE: Required for Val on samples. Not Req for Full-Inference, but DSC will be reported if provided.")
        logPrint("Filepaths to Ground-Truth labels of the Validation Cases = " + str(self.gtLabelsFilepathsVal))
        logPrint("Provided ROI masks for Validation = " + str(self.providedRoiMasksVal) + \
                 ". NOTE: Validation-sampling and Full-Inference will be limited within this mask if provided. \
                 If not provided, Negative Validation samples will be extracted from whole volume, \
                 except if advanced-sampling is enabled, and the user provided separate weight-maps for sampling.")
        logPrint("Filepaths to ROI masks for Validation Cases = " + str(self.roiMasksFilepathsVal))
        
        logPrint("~~~~~~~Validation on Samples throughout Training~~~~~~~")
        logPrint("Number of Segments loaded on GPU per subepoch for Validation = " + \
                 str(self.segmentsLoadedOnGpuPerSubepochVal))
        
        logPrint("~~Advanced Sampling~~")
        logPrint("Using default uniform sampling for validation = " + str(self.useDefaultUniformValidationSampling) + \
                 ". NOTE: Adv.Sampl.Params are auto-set to perform uniform-sampling if True.")
        logPrint("Type of Sampling = " + str(self.samplingTypeInstanceVal.getStringOfSamplingType()) + \
                 " ("+ str(self.samplingTypeInstanceVal.getIntSamplingType()) + ")")
        logPrint("Sampling Categories = " + str(self.samplingTypeInstanceVal.getStringsPerCategoryToSample()) )
        logPrint("Percent of Samples to extract per Sampling Category = " + \
                 str(self.samplingTypeInstanceVal.getPercentOfSamplesPerCategoryToSample()))
        logPrint("Provided Weight-Maps, pointing where to focus sampling for each category \
        (if False, samples will be extracted based on GT and ROI) = " + \
        str(self.providedWeightMapsToSampleForEachCategoryValidation))
        logPrint("Paths to weight-maps for sampling of each category = " + \
                 str(self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatValidation))
        
        logPrint("~~~~~Validation with Full Inference on Validation Cases~~~~~")
        logPrint("Perform Full-Inference on Val. cases every that many epochs = " + \
                 str(self.numberOfEpochsBetweenFullInferenceOnValImages))
        logPrint("~~Predictions (segmentations and prob maps on val. cases)~~")
        logPrint("Save Segmentations = " + str(self.saveSegmentationVal))
        logPrint("Save Probability Maps for each class = " + str(self.saveProbMapsBoolPerClassVal))
        logPrint("Filepaths to save results per case = " + str(self.filepathsToSavePredictionsForEachPatVal))
        logPrint("~~Feature Maps~~")
        logPrint("Save Feature Maps = " + str(self.saveIndividualFmImagesVal))
        logPrint("Save FMs in a 4D-image = " + str(self.saveMultidimImgWithAllFmsVal))
        logPrint("Min/Max Indices of FMs to visualise per pathway-type and per layer = " + \
                 str(self.indicesOfFmsToVisualisePerPathwayAndLayerVal))
        logPrint("Filepaths to save FMs per case = " + str(self.filepathsToSaveFeaturesForEachPatVal))
        
        logPrint("~~Optimization~~")
        logPrint("Initial Learning rate = " + str(self.learningRate))
        logPrint("Optimizer to use: SGD(0), Adam(1), RmsProp(2) = " + str(self.optimizerSgd0Adam1Rms2))
        logPrint("Parameters for Adam: b1= " + str(self.b1Adam) + ", b2=" + str(self.b2Adam) + ", e= " + str(self.eAdam) )
        logPrint("Parameters for RmsProp: rho= " + str(self.rhoRms) + ", e= " + str(self.eRms) )
        logPrint("Momentum Type: Classic (0) or Nesterov (1) = " + str(self.classicMom0Nesterov1))
        logPrint("Momentum Non-Normalized (0) or Normalized (1) = " + str(self.momNonNormalized0Normalized1))
        logPrint("Momentum Value = " + str(self.momentumValue))
        logPrint("~~L1/L2 Regularization~~")
        logPrint("L1 Regularization term = " + str(self.l1Reg))
        logPrint("L2 Regularization term = " + str(self.l2Reg))
        
        logPrint("~~Freeze Weights of Certain Layers~~")
        logPrint("Indices of layers from each type of pathway that will be kept fixed (first layer is 0):")
        logPrint("Normal pathway's layers to freeze = "+ str(self.indicesOfLayersPerPathwayTypeToFreeze[0]))
        logPrint("Subsampled pathway's layers to freeze = "+ str(self.indicesOfLayersPerPathwayTypeToFreeze[1]))
        logPrint("FC pathway's layers to freeze = "+ str(self.indicesOfLayersPerPathwayTypeToFreeze[2]))
        
        logPrint("~~~~~~~~~~~~~~~~~~Other Generic Parameters~~~~~~~~~~~~~~~~")
        logPrint("~~Pre Processing~~")
        logPrint("Pad Input Images = " + str(self.padInputImagesBool))
        
        logPrint("========== Done with printing session's parameters ==========")
        logPrint("=============================================================")
        
    def getTupleForCnnTraining(self) :
        borrowFlag = True
        
        trainTuple = (self.sessionLogger,
                self.pathAndFilenameToSaveTrainedModels,
                self.cnn3dInstance,
                
                self.performValidationOnSamplesThroughoutTraining,
                [self.saveSegmentationVal, self.saveProbMapsBoolPerClassVal],
                
                self.filepathsToSavePredictionsForEachPatVal,
                
                self.channelsFilepathsTrain,
                self.channelsFilepathsVal,
                
                self.gtLabelsFilepathsTrain,
                self.providedGtVal,
                self.gtLabelsFilepathsVal,
                #Always true, since either GT labels or advanced-mask-where-to-pos
                self.providedWeightMapsToSampleForEachCategoryTraining, 
                self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatTraining,
                #If false, corresponding samples will be extracted uniformly from whole image.
                self.providedWeightMapsToSampleForEachCategoryValidation, 
                self.forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatValidation,
                
                self.providedRoiMasksTrain, # also used for int-augm.
                self.roiMasksFilepathsTrain,# also used for int-augm
                self.providedRoiMasksVal, # also used for fast inf
                self.roiMasksFilepathsVal, # also used for fast inf and also for uniform sampling of segs.
                
                borrowFlag,
                self.numberOfEpochs,
                self.numberOfSubepochs,
                self.numOfCasesLoadedPerSubepoch,
                self.segmentsLoadedOnGpuPerSubepochTrain,
                self.segmentsLoadedOnGpuPerSubepochVal,
                
                #-------Sampling Type---------
                self.samplingTypeInstanceTrain,
                self.samplingTypeInstanceVal,
                
                #-------Preprocessing-----------
                self.padInputImagesBool,
                self.smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                #-------Data Augmentation-------
                self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd,
                self.reflectImagesPerAxis,
                
                self.useSameSubChannelsAsSingleScale,
                
                self.subsampledChannelsFilepathsTrain,
                self.subsampledChannelsFilepathsVal,
                
                #Learning Rate Schedule:
                self.stable0orAuto1orPredefined2orExponential3LrSchedule,
                self.minIncreaseInValidationAccuracyThatResetsWaiting,
                self.numEpochsToWaitBeforeLoweringLr,
                self.whenDecreasingDivideLrBy,
                self.predefinedSchedLowerLrAtEpochs,
                self.exponentialSchedForLrAndMom,
                
                #Weighting Classes differently in the CNN's cost function during training:
                self.numberOfEpochsToWeightTheClassesInTheCostFunction,
                
                self.performFullInferenceOnValidationImagesEveryFewEpochs, #Even if not providedGtForValidationBool, 
                #inference will be performed if this == True, to save the results, eg for visual.
                self.numberOfEpochsBetweenFullInferenceOnValImages, # Should not be == 0, except if 
                #performFullInferenceOnValidationImagesEveryFewEpochsBool == False
                
                #--------For FM visualisation---------
                self.saveIndividualFmImagesVal,
                self.saveMultidimImgWithAllFmsVal,
                self.indicesOfFmsToVisualisePerPathwayAndLayerVal,
                self.filepathsToSaveFeaturesForEachPatVal
                )
        return trainTuple
    
    
    def getTupleForInitializingTrainingState(self) :
        initializingTrainingStateTuple = (
                        self.sessionLogger,
                        
                        self.indicesOfLayersPerPathwayTypeToFreeze,
                        #=====COST FUNCTION=====
                        self.costFunctionLetter,
                        
                        #=====OPTIMIZATION=====
                        self.learningRate,
                        self.optimizerSgd0Adam1Rms2,
                        self.classicMom0Nesterov1, 
                        self.momentumValue,
                        self.momNonNormalized0Normalized1,
                        self.b1Adam,
                        self.b2Adam,
                        self.eAdam,
                        self.rhoRms,
                        self.eRms,
                        # Regularisation
                        self.l1Reg,
                        self.l2Reg
                        )
        
        return initializingTrainingStateTuple

    def getTupleForCompilationOfTrainFunc(self) :
        trainFunctionCompilationTuple = ( self.sessionLogger, )
        return trainFunctionCompilationTuple
       
    def getTupleForCompilationOfValFunc(self) :
        valFunctionCompilationTuple = ( self.sessionLogger, )
        return valFunctionCompilationTuple
       
    def getTupleForCompilationOfTestFunc(self) :
        testFunctionCompilationTuple = ( self.sessionLogger, )
        return testFunctionCompilationTuple



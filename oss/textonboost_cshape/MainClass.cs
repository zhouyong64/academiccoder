//********************************************************
//
// Jamie Shotton
// Machine Intelligence Laboratory
// Department of Engineering
// University of Cambridge, UK
// Copyright (c) 2006
// All Rights Reserved.
//
//  Permission is hereby granted, free of charge, to use and distribute
//  this software and its documentation without restriction, including
//  without limitation the rights to use, copy, modify, merge, publish,
//  distribute, sublicense, and/or sell copies of this work, and to
//  permit persons to whom this work is furnished to do so, subject to
//  the following conditions:
//   1. The code must retain the above copyright notice, this list of
//      conditions and the following disclaimer.
//   2. Any modifications must be clearly marked as such.
//   3. Original authors' names are not deleted.
//   4. The authors' names are not used to endorse or promote products
//      derived from this software without specific prior written
//      permission.
//
//  THE UNIVERSITY OF CAMBRIDGE AND THE CONTRIBUTORS TO THIS WORK
//  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
//  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
//  SHALL THE UNIVERSITY OF CAMBRIDGE NOR THE CONTRIBUTORS BE LIABLE
//  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
//  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
//  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
//  THIS SOFTWARE.
//
//********************************************************
//                      Author :  Jamie Shotton
//                      Date   :  May 2006
//  This work pertains to the research described in the ECCV 2006 paper
//  TextonBoost: Joint Appearance, Shape and Contex Modeling
//  for Multi-Class Object Recognition and Segmentation
//********************************************************



using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Text.RegularExpressions;

using Misc;
using Image;


namespace TextonBoost
{
    public class MainClass
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////// GLOBAL PARAMETERS /////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////// INITIALISATION PARAMETERS ////////////////////////////////////////////////////

        // Input file information
        public const string TempFolder = @"W:\temp\";
        public const string ImageFolder = @"W:\data\MSRC-2005\Sowerby\Images";
        public const string GroundTruthFolder = @"W:\data\MSRC-2005\Sowerby\GroundTruth";
        public static readonly List<string> ImageExtensions = new List<string>(new string[] { "png", "bmp", "jpg" });

        // Training / Validation / Test proportions - these should add to 1.0
        public const double ProportionTraining = 0.45;
        public const double ProportionValidation = 0.10;
        public const double ProportionTest = 0.45;

        // Filter Bank Rescale (multiplies the sigma of the filters from their original size given in Winn & Criminisi ICCV 2005)
        public const double FilterBankRescale = 0.7;


        ///////////////////////////////////////////////////// CLUSTERING PARAMETERS ////////////////////////////////////////////////////////

        public const int NumberOfClusters = 400;
        public const int ClusteringSubSample = 5; // Only take every nth pixel in x and y to cluster on (for efficiency)


        ///////////////////////////////////////////////////// BOOSTING PARAMETERS //////////////////////////////////////////////////////////

        // Number of rounds of boosting
        public const int NumRoundsBoosting = 700;

        // Randomization factor
        public const double RandomizationFactor = 0.003;

        // Sub-sampling factor
        public const int BoostingSubSample = 5;

        // Weak-learner threshold (theta) parameters
        public const int NumberOfThresholds = 25;
        public const int ThresholdStart = 5;
        public const int ThresholdIncrement = 40;

        // Shape parameters
        public const int NumberOfRectangles = 100;
        public const int MinimumRectangleSize = BoostingSubSample;  // Must be at least BoostingSubSample in size
        public const int MaximumRectangleSize = 200;


        ///////////////////////////////////////////////////// LIST OF CLASSES //////////////////////////////////////////////////////////

        // NB order of this list IS important (void=0, building=1, grass=2, etc) since it specifies the correspondence
        // to colour indexes in the ground truth data.
        // Also those names preceded with a "-" will be ignored for both training and testing purposes.
        public static readonly string[] Classes = new string[]
	        {
	        "-void",
	        "building",
	        "grass",
	        "tree",
	        "cow",
	        "-horse",
	        "sheep",
	        "sky",
	        "-mountain",
	        "aeroplane",
	        "water",
	        "face",
	        "car",
	        "bike",
	        "flower",
	        "sign",
	        "bird",
	        "book",
	        "chair",
	        "-motorbike",
	        "-person",
	        "road",
	        "cat",
	        "dog",
	        "body",
	        "boat"
	        };


        ///////////////////////////////////////////////////// EVALUATION PARAMETERS /////////////////////////////////////////////////////

        // Sub sampling
        public const int EvaluationSubSample = 1;

        // Test set (either Training, Test, or Validation)
        public const string TestSet = "Test";






        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////// MAIN METHOD /////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void Main(string[] args)
        {
            // Change this variable to perform training or testing
            bool train = true;

            if (train)
            {
                Initialise.DoWork();
                LearnTextonDictionary.DoWork();
                TextoniseImages.DoWork();
                new TextonBoost().DoWork();
            }
            else
            {
                Evaluate.DoWork();
            }
        }






        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////// UTILITY FIELDS AND METHODS //////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        public static readonly MyRandom Rnd = new MyRandom(0);

        private static string GetCompatibleName(string name)
        {
            Regex regex = new Regex(@"[\\:]");
            return regex.Replace(name, "-", int.MaxValue);
        }

        public static void SaveImage<T>(string name, Image<T> image, OutputMapping<T> mapping) where T : struct
        {
            string filename = TempFolder + @"\" + GetCompatibleName(name) + ".png";

            ImageIO.SaveImage<T>(filename, image, mapping);
        }

        public static void Save(string name, object data)
        {
            string filename = TempFolder + @"\" + GetCompatibleName(name) + ".data";

            using (FileStream outStream = new FileStream(filename, FileMode.Create))
            {
                BinaryFormatter fmt = new BinaryFormatter(null, new StreamingContext(StreamingContextStates.File));

                fmt.Serialize(outStream, data);
                outStream.Close();
            }
        }

        public static object Load(string name)
        {
            string filename = TempFolder + @"\" + GetCompatibleName(name) + ".data";

            using (Stream inStream = new FileStream(filename, FileMode.Open))
            {
                BinaryFormatter fmt = new BinaryFormatter();

                object data = fmt.Deserialize(inStream);
                inStream.Close();

                return data;
            }
        }
    }
}

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
using System.Text;

using Image;
using ImageProcessing;
using Misc;

namespace TextonBoost
{
    [Serializable]
    public struct ImageFilenamePair
    {
        public string im;
        public string gt;

        public ImageFilenamePair(string im) : this(im, null) { }
        public ImageFilenamePair(string im, string gt)
        {
            this.im = im;
            this.gt = gt;
        }
    }

    public class Initialise
    {
        #region Do Work
        private static readonly string[] splitNames = new string[] { "Training", "Validation", "Test" };
        private static readonly int numSplits = splitNames.Length;

        public static void DoWork()
        {
            // Compute train/validation/test split
            Console.WriteLine("Creating Training/Validation/Test Split...");
            List<ImageFilenamePair>[] split = ComputeSplit();

            // Save split
            for (int i = 0; i < numSplits; i++)
                MainClass.Save("Filenames." + splitNames[i], split[i]);

            // Calculate filter bank responses
            Console.WriteLine("Calculating Filter-Bank Responses...");
            ComputeFilterResponses(split);
        }

        // Compute filter bank responses for each image
        private static void ComputeFilterResponses(List<ImageFilenamePair>[] split)
        {
            // Create the filter-bank
            List<FilterBankElement> filterBank = CreateFilterBank(MainClass.FilterBankRescale);

            // Compute total number
            int total = 0;
            for (int splitIndex = 0; splitIndex < numSplits; splitIndex++)
                total += split[splitIndex].Count;

            for (int splitIndex = 0; splitIndex < numSplits; splitIndex++)
                for (int imNum = 0; imNum < split[splitIndex].Count; imNum++)
                {
                    string filename = split[splitIndex][imNum].im;

                    // Load the image (Lab)
                    Image<byte> image = ImageIO.LoadImage<byte>(filename);
                    Image<float> imLab = ColourConversion.RGBToLab<float>(Image<float>.CopyImage<byte>(image));

                    // Perform the convolutions
                    Image<float> filterResponses = new Image<float>(imLab.Size, filterBank.Count);
                    for (int filterIndex = 0; filterIndex < filterBank.Count; filterIndex++)
                    {
                        Kernel2D<float> filter = filterBank[filterIndex].filter;
                        int colourBand = filterBank[filterIndex].colourBand;
                        filter.Convolve(new VirtualImage<float>(filterResponses, filterIndex), new VirtualImage<float>(imLab, colourBand), ExtendMode.Extend);
                    }

                    MainClass.Save("FilterResponses." + filename, filterResponses);
                }
        }

        // Split the image data into training, validation and test sets
        private static List<ImageFilenamePair>[] ComputeSplit()
        {
            // Get parameter values
            string imageFolder = MainClass.ImageFolder;
            string groundTruthFolder = MainClass.GroundTruthFolder;
            List<string> imageExtensions = MainClass.ImageExtensions;
            double[] splitProportions = new double[numSplits];
            splitProportions[0] = MainClass.ProportionTraining;
            splitProportions[1] = MainClass.ProportionValidation;
            splitProportions[2] = MainClass.ProportionTest;

            // Enumerate images in folder
            List<ImageFilenamePair> filenames = new List<ImageFilenamePair>();
            DirectoryInfo d = new DirectoryInfo(imageFolder);
            FileInfo[] files = d.GetFiles();
            foreach (FileInfo file in files)
                if (imageExtensions.Contains(file.Extension.Substring(1)))
                    filenames.Add(new ImageFilenamePair(file.FullName));

            // Enumerate corresponding ground truth images
            for (int i = 0; i < filenames.Count; i++)
            {
                ImageFilenamePair ifp = filenames[i];
                string imName = Path.GetFileNameWithoutExtension(ifp.im);
                string imExt = Path.GetExtension(ifp.im);
                ifp.gt = groundTruthFolder + Path.DirectorySeparatorChar + imName + imExt;
                if (!File.Exists(ifp.gt))
                    throw new Exception("Missing ground truth file: " + ifp.gt);
                filenames[i] = ifp;
            }

            // Randomly permute
            for (int i = 0; i < filenames.Count * 5; i++)
            {
                int from = MainClass.Rnd.Next(filenames.Count);
                int to = MainClass.Rnd.Next(filenames.Count);

                ImageFilenamePair temp;

                temp = filenames[from];
                filenames[from] = filenames[to];
                filenames[to] = temp;
            }

            // Create Train/Validation/Test split
            List<ImageFilenamePair>[] split = new List<ImageFilenamePair>[numSplits];
            for (int i = 0; i < numSplits; i++) split[i] = new List<ImageFilenamePair>();
            // Make splitProportions cumulative
            for (int i = 1; i < numSplits; i++)
                splitProportions[i] = splitProportions[i] + splitProportions[i - 1];

            int splitIndex = 0;
            for (int i = 0; i < filenames.Count; i++)
            {
                // Update splitIndex
                for (; splitIndex < numSplits; splitIndex++)
                    if (i / (double)filenames.Count < splitProportions[splitIndex])
                        break;
                if (splitIndex >= numSplits)
                    break;

                // Add to split
                split[splitIndex].Add(filenames[i]);
            }
            

            Console.WriteLine("Split works out as:");
            for (splitIndex = 0; splitIndex < numSplits; splitIndex++)
            {
                Console.WriteLine(Environment.NewLine + "splitIndex = " + splitIndex + ", count = " + split[splitIndex].Count);
                foreach (ImageFilenamePair ifp in split[splitIndex])
                    Console.WriteLine(ifp.im + " <=> " + ifp.gt);
            }

            return split;
        }

        #endregion

        #region Filter Bank

        private class FilterBankElement
        {
            public Kernel2D<float> filter;
            public int colourBand;
            public FilterBankElement(Kernel2D<float> filter, int colourBand)
            {
                this.filter = filter;
                this.colourBand = colourBand;
            }
        }

        private static List<FilterBankElement> CreateFilterBank(double rescale)
        {
            // Create filterBank
            List<FilterBankElement> filterBank = new List<FilterBankElement>();
            
            // Gaussians (applied to all colour channels)
            for (int band = 0; band < 3; band++)
                for (double sigma = 1.0; sigma <= 4.0; sigma *= 2.0)
                    filterBank.Add(new FilterBankElement(Kernel2D<float>.CreateGaussian(sigma * rescale), band));
 
            // Laplacians (applied to just greyscale)
            for (double sigma = 1.0; sigma <= 8.0; sigma *= 2.0)
                filterBank.Add(new FilterBankElement(Kernel2D<float>.CreateLaplacian(sigma * rescale), 0));

            // Derivatives of Gaussians (appiled to just greyscale)
            for (double sigma = 2.0; sigma <= 4.0; sigma *= 2.0)
            {
                filterBank.Add(new FilterBankElement(Kernel2D<float>.CreateGaussianDerivativeX(sigma * rescale, 3.0 * sigma * rescale), 0)); // d/dx
                filterBank.Add(new FilterBankElement(Kernel2D<float>.CreateGaussianDerivativeY(3.0 * sigma * rescale, sigma * rescale), 0)); // d/dy
            }

            return filterBank;
        }



        #endregion
    }
}

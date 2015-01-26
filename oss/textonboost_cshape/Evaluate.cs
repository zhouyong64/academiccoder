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
using System.Drawing;
using System.IO;
using System.Text;
using System.Threading;

using Algorithms;
using Image;
using ImageProcessing;
using Misc;

namespace TextonBoost
{
    public class Evaluate
    {
        // Other parameters
        private static int SubSample;
        private static int NumRounds;
        private static string TestSet;

        public static void DoWork()
        {
            // Get parameters
            SubSample = MainClass.EvaluationSubSample;
            NumRounds = MainClass.NumRoundsBoosting;
            TestSet = MainClass.TestSet;

            // Get list of test data
            List<ImageFilenamePair> testDataFilenames = (List<ImageFilenamePair>) MainClass.Load("Filenames." + TestSet);

            // Load shape filters
            List<TextonBoost.ShapeFilter> shapeFilters = new List<TextonBoost.ShapeFilter>();
            for (int round = 0; round < NumRounds; round++)
                shapeFilters.Add((TextonBoost.ShapeFilter) MainClass.Load("ShapeFilter." + round));

            TextonBoost.ClassList classList = shapeFilters[0].classList;

            // Get number of textons
            List<KMeans.Cluster> clusters = (List<KMeans.Cluster>) MainClass.Load("Clusters");
            int numTextons = clusters.Count;

            for (int i = 0; i < testDataFilenames.Count; i++)
            {
                Console.WriteLine("Loading images " + i);
                Image<byte> image = ImageIO.LoadImage<byte>(testDataFilenames[i].im);
                Image<int> textonMap = (Image<int>) MainClass.Load("TextonMaps." + testDataFilenames[i].im);

                Console.WriteLine("  Classifying...");
                NormalisingOutputMapping<double> normMap = new NormalisingOutputMapping<double>();
                Image<double>[] classifications = ClassifyImage(textonMap, classList, shapeFilters, NumRounds, numTextons, SubSample);

                // Compute mode
                Image<int> mode = GetMAP(classifications[classifications.Length - 1]);

                // Save
                MainClass.SaveImage(testDataFilenames[i].im, mode, new IdOutputMapping<int>());


            }
        }

        private static unsafe Image<double>[] ClassifyImage(Image<int> textonMap, TextonBoost.ClassList classList, List<TextonBoost.ShapeFilter> shapeFilters, int step, int numTextons, int subSample)
        {
            int ssW = (textonMap.Width + subSample - 1) / subSample; // Care with off by one errors
            int ssH = (textonMap.Height + subSample - 1) / subSample; // Care with off by one errors

            // Compute integral images
            Image<int>[] integralImages = TextonBoost.CalculateIntegralImages(textonMap, numTextons, subSample, new Size(ssW, ssH));

            List<Image<double>> classificationFullSizes = new List<Image<double>>();
            Image<double> classification = new Image<double>(ssH, ssW, classList.NumCompact);

            fixed (double* classificationBase = classification.DataPointer)
            {
                // Iteratively classify
                for (int round = 0; round < shapeFilters.Count; round++)
                {
                    TextonBoost.ShapeFilter sf = shapeFilters[round];

                    Rectangle r = TextonBoost.SubSample(sf.r, subSample);

                    double* classificationPtr = classificationBase;
                    foreach (Point p in classification.Bounds)
                    {
                        double response = TextonBoost.CalculateShapeFilterResponse(integralImages[sf.t], r, p);

                        double confidenceShared = (response > sf.wl.theta) ? (sf.wl.a + sf.wl.b) : sf.wl.b;

                        for (int c = 0; c < classList.NumCompact; c++, classificationPtr++)
                            *classificationPtr += sf.wl.n.Contains(c) ? confidenceShared : sf.wl.k[c];
                    }

                    if (round % step == step - 1)
                    {
                        // Convert to pdf (soft-max function)
                        Image<double> classificationPDF = new Image<double>(ssH, ssW, classList.NumCompact);
                        foreach (Point p in classification.Bounds)
                        {
                            double sum = 0.0;
                            for (int c = 0; c < classList.NumCompact; c++)
                                sum += Math.Exp(classification[p, c]);

                            for (int c = 0; c < classList.NumCompact; c++)
                                classificationPDF[p, c] = Math.Exp(classification[p, c]) / sum;
                        }

                        // Super-sample (spatially and compact to full)
                        Image<double> classificationFullSize = new Image<double>(textonMap.Size, classList.NumFull);
                        foreach (Point p in classificationFullSize.Bounds)
                            for (int c = 0; c < classList.NumCompact; c++)
                                classificationFullSize[p, classList.ToFull(c)] = classificationPDF[p.Y / subSample, p.X / subSample, c];

                        classificationFullSizes.Add(classificationFullSize);
                    }
                }
            }

            return classificationFullSizes.ToArray();
        }

        public static Image<double> SubSampleClassification(Image<double> classification, int subSample)
        {
            int ssW = (classification.Width + subSample - 1) / subSample; // Care with off by one errors
            int ssH = (classification.Height + subSample - 1) / subSample; // Care with off by one errors

            Image<double> ssClassification = new Image<double>(ssH, ssW, classification.Bands);

            foreach (Point p in ssClassification.Bounds)
                for (int c = 0; c < classification.Bands; c++)
                    ssClassification[p, c] = classification[p.Y * subSample, p.X * subSample, c];

            return ssClassification;
        }

        public static Image<double> SuperSampleClassification(Image<double> ssClassification, int subSample, Size origSize)
        {
            Image<double> classification = new Image<double>(origSize, ssClassification.Bands);

            foreach (Point p in classification.Bounds)
                for (int c = 0; c < classification.Bands; c++)
                    classification[p, c] = ssClassification[p.Y / subSample, p.X / subSample, c];

            return classification;
        }

        #region Visualisations

        public static Image<int> GetMAP(Image<double> classification)
        {
            Image<int> map = new Image<int>(classification.Size);

            foreach (Point p in classification.Bounds)
            {
                int max = 0;
                double maxP = classification[p, max];
                for (int c = 1; c < classification.Bands; c++)
                    if (classification[p, c] > maxP)
                    {
                        max = c;
                        maxP = classification[p, c];
                    }
                map[p] = max;
            }

            return map;
        }

        public static Image<double> GetEntropy(Image<double> classification)
        {
            Image<double> outImg = new Image<double>(classification.Size);

            foreach (Point p in classification.Bounds)
            {
                double entropy = 0.0;
                for (int c = 0; c < classification.Bands; c++)
                {
                    double prob = classification[p, c];
                    if (prob > 0.0)
                        entropy -= prob * Math.Log(prob);
                }

                outImg[p] = entropy / Math.Log(2.0);
            }

            return outImg;
        }

        private static Image<byte> GetEntropyVis(Image<int> classificationMAP, Image<double> entropy, int numClasses)
        {
            Image<byte> outImg = new Image<byte>(classificationMAP.Size, 3);

            double norm = Math.Log(numClasses) / Math.Log(2.0);

            foreach (Point p in outImg.Bounds)
            {
                Color outColour = IdOutputMapping<byte>.MapToColor(classificationMAP[p]);

                double scale = (1.0 - entropy[p] / norm);
                outImg[p, 0] = (byte)(outColour.R * scale);
                outImg[p, 1] = (byte)(outColour.G * scale);
                outImg[p, 2] = (byte)(outColour.B * scale);
            }

            return outImg;
        }

        #endregion
    }
}

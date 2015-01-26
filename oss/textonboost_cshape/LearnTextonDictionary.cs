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
using System.Text;

using Algorithms;
using Image;

namespace TextonBoost
{
    class LearnTextonDictionary
    {
        public static void DoWork()
        {
            // Get Parameters
            List<ImageFilenamePair> trainingDataFilenames = (List<ImageFilenamePair>) MainClass.Load("Filenames.Training");
            int K = MainClass.NumberOfClusters;
            int SubSample = MainClass.ClusteringSubSample;

            // Load images and calculate total number of pixels
            int D = 0;
            int N = 0;
            double[][][] dataTemp = new double[trainingDataFilenames.Count][][];
            for (int i = 0; i < trainingDataFilenames.Count; i++)
            {
                Console.WriteLine("Loading filter responses " + i);
                ImageFilenamePair ifp = trainingDataFilenames[i];
                Image<float> temp = (Image<float>) MainClass.Load("FilterResponses." + ifp.im);
                int tempN = ((temp.Width + (SubSample - 1)) / SubSample) * ((temp.Height + (SubSample - 1)) / SubSample);
                D = temp.Bands;

                // Sub-sample data
                dataTemp[i] = new double[tempN][];
                int nTemp = 0;
                foreach (Point p in temp.Bounds)
                {
                    if (p.X % SubSample != 0 || p.Y % SubSample != 0) continue;
                    dataTemp[i][nTemp] = new double[D];
                    for (int d = 0; d < D; d++)
                        dataTemp[i][nTemp][d] = (double)temp[p, d];
                    nTemp++;
                }

                N += tempN;
            }

            // Move to big array
            Console.WriteLine("Initialising K-Means");
            double[][] data = new double[N][];
            int n = 0;
            for (int i = 0; i < trainingDataFilenames.Count; i++)
                for (int j = 0; j < dataTemp[i].Length; j++)
                    data[n++] = dataTemp[i][j];

            // Cluster
            KMeans kmeans = new KMeans(data, K, new KMeans.EuclideanDistanceMeasure(), 0.01);

            int iter = 0;
            while (!kmeans.Converged)
            {
                Console.WriteLine("Iteration " + (iter++) + "...");
                kmeans.PerformIteration();
            }

            // Save result
            MainClass.Save("Clusters", kmeans.GetClusters());
        }
    }
}

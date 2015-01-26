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
    public class TextoniseImages
    {
        private const int PointsPerKDTreeCluster = 30;

        public static void DoWork()
        {
            List<ImageFilenamePair> filenames = new List<ImageFilenamePair>();
            foreach (string s in new string[] { "Training", "Validation", "Test" })
                filenames.AddRange((List<ImageFilenamePair>) MainClass.Load("Filenames." + s));

            // Load clusters and build kd-tree
            List<KMeans.Cluster> clusters = (List<KMeans.Cluster>) MainClass.Load("Clusters");
            double[][] data = new double[clusters.Count][];
            for (int i = 0; i < data.Length; i++)
                data[i] = clusters[i].Mean;
            kdTree kd = new kdTree(data, PointsPerKDTreeCluster);

            for (int i = 0; i < filenames.Count; i++)
            {
                Console.WriteLine("Textonising " + i);

                ImageFilenamePair ifp = filenames[i];
                Image<float> filterResponses = (Image<float>) MainClass.Load("FilterResponses." + ifp.im);
                Image<int> textonMap = new Image<int>(filterResponses.Size);
                foreach (Point p in filterResponses.Bounds)
                {
                    double[] point = new double[filterResponses.Bands];
                    for (int d = 0; d < filterResponses.Bands; d++)
                        point[d] = filterResponses[p, d];

                    textonMap[p] = kd.NearestNeighbour(point);
                }

                MainClass.Save("TextonMaps." + ifp.im, textonMap);
            }
        }
    }
}

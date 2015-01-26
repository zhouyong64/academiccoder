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
using System.Text;

using Image;
using Misc;

namespace Algorithms
{
    public class KMeans
    {
        public abstract class DistanceMeasure
        {
            public abstract double Distance(Cluster cluster, double[] x);
            public abstract double Distance(Cluster cluster, double[] x, double minDistance);
        }

        public class EuclideanDistanceMeasure : DistanceMeasure
        {
            public override double Distance(Cluster cluster, double[] x)
            {
                int D = cluster.mean.Length;

                double euclideanDistance = 0.0;
                for (int d = 0; d < D; d++)
                {
                    double diff = x[d] - cluster.mean[d];
                    euclideanDistance += diff * diff;
                }
                return Math.Sqrt(euclideanDistance);
            }

            public override double Distance(Cluster cluster, double[] x, double minDistance)
            {
                int D = cluster.mean.Length;
                minDistance = minDistance * minDistance; // Square minDistance so can be compared without square-rooting

                double euclideanDistance = 0.0;
                for (int d = 0; d < D; d++)
                {
                    double diff = x[d] - cluster.mean[d];
                    euclideanDistance += diff * diff;
                    if (euclideanDistance > minDistance) return double.PositiveInfinity;
                }
                return Math.Sqrt(euclideanDistance);
            }
        }

        // Cluster type
        [Serializable]
        public struct Cluster
        {
            internal double[] mean, invCov;
            internal double logDetCov, logMixCoeff;
            internal int count;

            internal bool Valid
            {
                get { return count > 0; }
            }

            public double[] Mean
            {
                get { return mean; }
            }

            public double[] Covariance
            {
                get
                {
                    double[] covariance = new double[invCov.Length];
                    for (int i = 0; i < covariance.Length; i++)
                        covariance[i] = 1.0 / invCov[i];
                    return covariance;
                }
            }
        }

        public enum InitType {Random}; // Initialisation Type

        private static readonly double logTwoPi = Math.Log(2.0 * Math.PI);
        private static readonly Random rnd = new Random(0);

        private DistanceMeasure distanceMeasure; // Distance measure to use
        private int K; // Number of clusters
        private int N; // Number of data points
        private int D; // Dimensionality of data points

        private double[][] data; // Data points
        private int[] labels; // Labels
        private Cluster[] clusters; // Clusters

        // Change variables
	    private double currentChange;
	    private double maxChange;

        #region Initialisation

        public KMeans(double[][] data, int K, DistanceMeasure distanceMeasure, double maxChange) : this(data, K, distanceMeasure, maxChange, InitType.Random) { }

        public KMeans(double[][] data, int K, DistanceMeasure distanceMeasure, double maxChange, InitType init)
        {
            this.data = data;
            this.K = K;
            this.distanceMeasure = distanceMeasure;
            this.maxChange = maxChange;

            N = data.Length;
            D = data[0].Length;

            // Allocate memory
            labels = new int[N];
            clusters = new Cluster[K];
            for (int k = 0; k < K; k++)
            {
                clusters[k].mean = new double[D];
                clusters[k].invCov = new double[D];
            }

            switch (init)
            {
                case InitType.Random:
                    InitRandom();
                    break;
            }

            currentChange = 1.0;
        }

        private void InitRandom()
        {
            for (int k = 0; k < K; k++)
            {
                int i = rnd.Next(N);
                double detCov = 1.0;
                for (int d = 0; d < D; d++)
                {
                    clusters[k].mean[d] = data[i][d];
                    clusters[k].invCov[d] = 1.0;
                    detCov *= clusters[k].invCov[d];
                }
                clusters[k].logDetCov = (detCov > 0) ? Math.Log(detCov) : double.NegativeInfinity;
                clusters[k].logMixCoeff = Math.Log(1.0 / K);
                clusters[k].count = 10; // Faux value for initialisation
            }
            AssignLabels(data, clusters, labels);
        }

        #endregion

        #region Iteration

        private double AssignLabels(double[][] data, Cluster[] clusters, int[] labels)
        {
            // Initialise
            int N = data.Length;
            int D = data[0].Length;
            int K = clusters.Length;
            double[,] accelerationMatrix = new double[K, K];
            double[] sVec = new double[K], uVec = new double[N];
            for (int i = 0; i < N; i++)
                uVec[i] = double.PositiveInfinity;
   
           	// Acceleration: Compute distances between all pairs of centres
	        for(int k1=0; k1<K; k1++)
	        {
		        if (!clusters[k1].Valid) continue;
		        for(int k2=0; k2<K; k2++)
		        {
			        if (!clusters[k2].Valid) continue;
			        accelerationMatrix[k1,k2] = distanceMeasure.Distance(clusters[k2], clusters[k1].mean);
		        }
	        }

	        // Acceleration: Compute min distances of each centre to all others
	        for(int k1=0; k1<K; k1++)
	        {
		        if (!clusters[k1].Valid) continue;
		        sVec[k1] = double.PositiveInfinity;
		        for(int k2=0; k2<K; k2++)
		        {
			        if (k1==k2 || !clusters[k2].Valid) continue;
			        double d = accelerationMatrix[k1,k2];
			        if (d<sVec[k1])
				        sVec[k1]=d;
		        }
		        sVec[k1] /= 2.0;
            }

	        // Assignment update (accelerated)
	        int changedCount = 0;
	        for(int i=0;i<N;i++)
	        {
		        int kOld = labels[i];
		        if (uVec[i] <= sVec[kOld])  // Lemma 2
			        continue;

		        double currentDist = distanceMeasure.Distance(clusters[kOld], data[i]);
		        uVec[i] = currentDist;

		        if (currentDist <= sVec[kOld]) // Lemma 2
			        continue;

		        bool changed = false;
		        for(int kNew=0; kNew<K; kNew++)
		        {
			        if (kNew==kOld || !clusters[kNew].Valid || accelerationMatrix[kOld,kNew] >= 2*currentDist) continue;
                    double dist = distanceMeasure.Distance(clusters[kNew], data[i], currentDist);
			        if (dist<currentDist)
			        {
				        labels[i] = kNew;
				        currentDist = dist;
				        uVec[i] = currentDist;
				        changed = true;
			        }
		        }
		        if (changed)
			        changedCount++;
	        }

	        return changedCount / (double) N;
        }

        private void ReestimateClusters()
        {
	        // Compute means
	        for(int k=0;k<K;k++)
	        {
                clusters[k].count = 0;
		        for(int d=0;d<D;d++)
			        clusters[k].mean[d] = 0.0;
	        }

            int sumCounts = 0;
	        for(int i=0;i<N;i++)
	        {
		        int k = labels[i];
		        for(int d=0;d<D;d++)
			        clusters[k].mean[d] += data[i][d];
		        clusters[k].count ++;
                sumCounts++;
	        }

	        for(int k=0;k<K;k++)
	        {
		        if (!clusters[k].Valid) continue;
		        for(int d=0;d<D;d++)
			        clusters[k].mean[d] /= clusters[k].count;
	        }

	        // Compute covariances
	        for(int k=0;k<K;k++)
	        {
                if (!clusters[k].Valid) continue;
		        for(int d=0;d<D;d++)
			        clusters[k].invCov[d] = 0.0;
	        }

	        for(int i=0;i<N;i++)
	        {
		        int k = labels[i];
		        for(int d=0;d<D;d++)
		        {
			        double diff = data[i][d] - clusters[k].mean[d];
                    clusters[k].invCov[d] += diff * diff;
		        }
	        }

	        for(int k=0;k<K;k++)
	        {
                if (!clusters[k].Valid) continue;

                double detCov = 1.0;
                for (int d = 0; d < D; d++)
                {
                    if (clusters[k].invCov[d] == 0.0)
                    {
                        //throw new Exception("Cluster covariance zero!!");
                        clusters[k].count = 0; // Force invalid cluster
                        break;
                    }

                    clusters[k].invCov[d] = clusters[k].count / clusters[k].invCov[d];
                    detCov /= clusters[k].invCov[d];
                }

                double mixCoeff = (double) clusters[k].count / (double) sumCounts;

                clusters[k].logDetCov = (detCov > 0) ? Math.Log(detCov) : double.NegativeInfinity;
                clusters[k].logMixCoeff = (mixCoeff > 0) ? Math.Log(mixCoeff) : double.NegativeInfinity;
	        }
        }

        public void PerformIteration()
        {
            ReestimateClusters();
            currentChange = AssignLabels(data, clusters, labels);
        }
        
        public bool Converged
        {
            get { return currentChange <= maxChange; }
        }

        #endregion

        #region Results

        public List<Cluster> GetClusters()
        {
            List<Cluster> result = new List<Cluster>();
            for (int k = 0; k < K; k++)
            {
                if (!clusters[k].Valid) continue;
                result.Add(clusters[k]);
            }

            return result;
        }

        public Image<int> GetVisualisation() // Only first two dimensions
        {
	        double minX=double.PositiveInfinity, maxX=double.NegativeInfinity, minY=double.PositiveInfinity, maxY=double.NegativeInfinity;

	        for(int i=0;i<N;i++)
	        {
		        minX = Math.Min(minX, data[i][0]);
		        minY = Math.Min(minY, data[i][1]);
		        maxX = Math.Max(maxX, data[i][0]);
		        maxY = Math.Max(maxY, data[i][1]);
	        }

	        int imgSize = 400;
            Image<int> img = new Image<int>(imgSize, imgSize);
	        for(int i=0;i<N;i++)
	        {
                int label = labels[i];
                if (!clusters[label].Valid)
                    continue;
                int x = Math.Max(0, Math.Min(imgSize - 1, (int)((data[i][0] - minX) / (maxX - minX) * imgSize)));
                int y = Math.Max(0, Math.Min(imgSize - 1, (int)((data[i][1] - minY) / (maxY - minY) * imgSize)));
                img[y,x] = label + 1;
	        }

            return img;
        }

        #endregion

    }
}

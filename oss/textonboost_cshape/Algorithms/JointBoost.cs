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

using Misc;

namespace Algorithms
{
    public class JointBoost
    {
        public interface IBoost
        {
            void Initialise();

            int N { get; }
            int D { get; }
            int C { get; }
            int ThetaStart { get; }
            int ThetaInc { get; }
            int NumThetas { get; }
            double AcceptFraction { get; }

            void CalculateFeatureValues(int[] values, int d);
            void GetTargets(int[] targets);
        }

        [Serializable]
        public struct SharingSet
        {
            private int n;

            public SharingSet(SharingSet from, int cNew)
            {
                n = from.n;
                n |= (1 << cNew);
            }

            public bool IsEmpty
            {
                get { return n == 0; }
            }
            
            public bool Contains(int c)
            {
                return (n & (1 << c)) != 0;
            }
        }

        [Serializable]
        public struct WeakLearner
        {
            public int d; // Feature number
            public double error; // Training error
            public double a, b; // Confidence weights
            public int theta; // Threshold
            public SharingSet n; // Sharing set
            public double[] k; // Constants for classes not in the sharing set
            public double timeTaken; // Time taken to optimise this weak learner (milliseconds)
        }


        private IBoost iBoost;
        private Random rnd;

        private readonly int N; // Num Examples
        private readonly int D; // Num Features
        private readonly int C; // Num Classes
        private readonly int thetaStart, thetaInc;
        private int[] thetaV; // Thresholds
        private double[,] weights; // Weights
        private int[] targets; // Target values

        // Scratch-space
        private double[] pV, tV, kV;
        private int[] featureValues;
        private double[,] qV, uV;

        public JointBoost(IBoost iBoost)
        {
            this.iBoost = iBoost;
            rnd = new Random(0);  // Start from same random seed each time

            iBoost.Initialise();

            this.N = iBoost.N;
            this.D = iBoost.D;
            this.C = iBoost.C;

            // Initialise thresholds
            thetaStart = iBoost.ThetaStart;
            thetaInc = iBoost.ThetaInc;
            thetaV = new int[iBoost.NumThetas];
            for (int t = 0; t < thetaV.Length; t++)
                thetaV[t] = thetaStart + t * thetaInc;

            // Initialise weights and targets
            targets = new int[N];
            iBoost.GetTargets(targets);
            weights = new double[N, C];
            for (int i = 0; i < N; i++)
                for (int c = 0; c < C; c++)
                    weights[i, c] = 1.0;

            // Initialise scratch space
            pV = new double[C];
            tV = new double[C];
            kV = new double[C];
            featureValues = new int[N];
            qV = new double[thetaV.Length, C];
            uV = new double[thetaV.Length, C];
        }

        public WeakLearner PerformRound()
        {
            DateTime start = DateTime.Now;

            // Minimum search variables
            WeakLearner minimum = new WeakLearner(); minimum.error = double.PositiveInfinity;

            // Precompute vectors pV, tV, kV
            CalculateVectorsPTK();

            for (int d = 0; d < D; d++)
            {
                // Randomisation
                if (rnd.NextDouble() > iBoost.AcceptFraction) continue;

                // Calculate feature values
                iBoost.CalculateFeatureValues(featureValues, d);

                // Generate sums using histograms
                CalculateVectorsQU(featureValues, qV, uV);

                // Optimise over sharing
                WeakLearner optimalSharing = OptimiseSharing(qV, uV);

                if (optimalSharing.error < minimum.error)
                {
                    minimum = optimalSharing;
                    minimum.d = d; // Store the feature number
                }
            }

            if (minimum.error != double.PositiveInfinity)
                UpdateWeights(minimum, featureValues);
            //else
            //    throw new Exception("Failed to find improved weak learner!");

            minimum.timeTaken = (DateTime.Now - start).TotalMilliseconds;

            return minimum;
        }

        private void CalculateVectorsPTK()
        {
            for (int c = 0; c < C; c++)
            {
                pV[c] = tV[c] = 0;
                for (int i = 0; i < N; i++)
                {
                    pV[c] += weights[i, c];
                    tV[c] += weights[i, c] * (targets[i] == c ? 1.0 : -1.0);
                }
                kV[c] = tV[c] / pV[c];
            }
        }

        private unsafe void CalculateVectorsQU(int[] featureValues, double[,] qV, double[,] uV)
        {
            fixed (int* fvBase = featureValues, tBase = targets)
            fixed (double* qVBase = qV, uVBase = uV, wBase = weights)
            {
                int* fvPtr = fvBase, tPtr = tBase;
                double* qVPtr = qVBase, uVPtr = uVBase;
                double* wPtr = wBase;

                // Initialise
                for (int tc = 0; tc < thetaV.Length * C; tc++, qVPtr++, uVPtr++)
                {
                    *qVPtr = 0;
                    *uVPtr = 0;
                }

                // Every example votes
                for (int i = 0; i < N; i++, fvPtr++, tPtr++)
                {
                    // Calculate bin number
                    int t = (*fvPtr - thetaStart) / thetaInc; //(int)((*fvPtr - thetaStart) / thetaInc);

                    if (t > thetaV.Length - 1) // Value too large - falls in final bin
                        t = thetaV.Length - 1;
                    else if (t < 0) // Value too small - has no influence
                        continue;

                    int targetVal = *tPtr;

                    qVPtr = qVBase + t * C; uVPtr = uVBase + t * C;
                    for (int c = 0; c < C; c++, qVPtr++, uVPtr++, wPtr++)
                    {
                        *qVPtr += *wPtr;
                        *uVPtr += *wPtr * (targetVal == c ? 1.0 : -1.0);
                    }
                }

                // Compute cumulative histograms
                qVPtr = qVBase + (thetaV.Length - 2) * C; uVPtr = uVBase + (thetaV.Length - 2) * C;
                for (int t = thetaV.Length - 2; t >= 0; t--, qVPtr -= 2*C, uVPtr -= 2*C)
                    for (int c = 0; c < C; c++, qVPtr++, uVPtr++)
                    {
                        *qVPtr += *(qVPtr + C);
                        *uVPtr += *(uVPtr + C);
                    }
            }
        }

        // Non-optimised version
/*        private void CalculateVectorsQU(double[] featureValues)
        {
            // Initialise
            for (int t = 0; t < thetaV.Length; t++)
                for (int c = 0; c < C; c++)
                {
                    qV[t, c] = 0.0;
                    uV[t, c] = 0.0;
                }

            // Every example votes
            for (int i = 0; i < N; i++)
            {
                int t = Math.Min(thetaV.Length - 1, (int)((featureValues[i] - thetaStart) / thetaInc));

                if (t < 0) // Value too small - has no influence
                    continue;

                for (int c = 0; c < C; c++)
                {
                    qV[t, c] += weights[i, c];
                    uV[t, c] += weights[i, c] * targets[i, c];
                }
            }

            // Compute cumulative histograms
            for (int t = thetaV.Length - 2; t >= 0; t--)
                for (int c = 0; c < C; c++)
                {
                    qV[t, c] += qV[t + 1, c];
                    uV[t, c] += uV[t + 1, c];
                }
        }*/


        private WeakLearner OptimiseSharing(double[,] qV, double[,] uV)
        {
            // Quadratic O(C^2) cost algorithm for searching over possible sharing sets (see Torralba et al)

            SharingSet mask = new SharingSet();
            WeakLearner[] testWL = new WeakLearner[C];
            for (int nBits = 0; nBits < C; nBits++)
                testWL[nBits].error = double.PositiveInfinity;
            SharingSet[] tempN = new SharingSet[C];

            //// Try sharing between zero classes (i.e. only ks used)
            //WeakLearner optimalWLZero = OptimiseWeakLearner(new SharingSet(), qV, uV);
            //if (optimalWLZero.error > 0 && optimalWLZero.error < testWL[0].error)
            //    testWL[0] = optimalWLZero;

            // Optimise over number of bits allowed
            for (int nBits = 1; nBits < C; nBits++)
            {
                List<SharingSet> allowed = new List<SharingSet>();
                for (int bit = 0; bit < C; bit++)
                {
                    if (mask.Contains(bit)) continue;

                    allowed.Add(new SharingSet(mask, bit));
                }

                foreach (SharingSet n in allowed)
                {
                    WeakLearner optimalWL = OptimiseWeakLearner(n, qV, uV);
                    if (optimalWL.error > 0 && optimalWL.error < testWL[nBits].error)
                    {
                        testWL[nBits] = optimalWL;
                        tempN[nBits] = n;
                    }
                }

                mask = tempN[nBits];
            }

            // Choose nBits with lowest error
            WeakLearner min = testWL[1];
            for (int nBits = 2; nBits < C; nBits++)
                if (testWL[nBits].error < min.error)
                    min = testWL[nBits];

            return min;
        }

        private WeakLearner OptimiseWeakLearner(SharingSet n, double[,] qV, double[,] uV)
        {
            WeakLearner min = new WeakLearner();
            min.n = n;
            min.k = new double[C];
            min.error = double.PositiveInfinity;

            for (int tIndex = 0; tIndex < thetaV.Length; tIndex++)
            {
                // Compute particular values of p, q, t, u
                double p = 0.0, q = 0.0, t = 0.0, u = 0.0;
                for(int c=0;c<C;c++)
                    if (n.Contains(c))
                    {
                        p += pV[c];
                        q += qV[tIndex, c];
                        t += tV[c];
                        u += uV[tIndex, c];
                    }

                double a = 0.0, b = 0.0;

                if (/*!n.IsEmpty && */!SolveMatrix(p, q, t, u, out a, out b))
                    continue;

                // Compute Jwse for this a, b, theta
                double Jwse = 0.0;
                for (int c = 0; c < C; c++)
                {
                    if (n.Contains(c))
                        Jwse += pV[c] - 2.0 * a * uV[tIndex, c] - 2.0 * b * tV[c] +
                                                a * a * qV[tIndex, c] + b * b * pV[c] +
                                                2.0 * a * b * qV[tIndex, c];
                    else
                        Jwse += pV[c] - 2.0 * kV[c] * tV[c] + kV[c] * kV[c] * pV[c];
                }

                // If improves then store weak learner
                if (Jwse >= 0 && Jwse < min.error)
                {
                    min.error = Jwse;
                    min.theta = thetaV[tIndex];
                    min.a = a;
                    min.b = b;
                    for (int c = 0; c < C; c++)
                        min.k[c] = n.Contains(c) ? 0.0 : kV[c];
                }
            }

            return min;
        }

        private static bool SolveMatrix(double p, double q, double t, double u, out double a, out double b)
        {
            const double tol = 1e-6;

            double pq = p - q;
            double tu = t - u;

            a = 0.0; b = 0.0;

            if (Math.Abs(q) < tol || Math.Abs(pq) < tol)
                return false;

            b = tu / pq;
            a = (u / q) - b;

            return true;
        }

        public void ContinueFrom(List<WeakLearner> wls)
        {
            foreach (WeakLearner wl in wls)
                UpdateWeights(wl, featureValues);
        }

        private void UpdateWeights(WeakLearner wl, int[] featureValues)
        {
            // Calculate feature values
            iBoost.CalculateFeatureValues(featureValues, wl.d);

            // Update the weights
            for (int i = 0; i < N; i++)
            {
                // Calculate confidence value h_m for those classes in sharing set
                double confidence = (featureValues[i] > wl.theta) ? (wl.a + wl.b) : wl.b;

                for (int c = 0; c < C; c++)
                    weights[i, c] *= Math.Exp(-(targets[i] == c ? 1.0 : -1.0) * (wl.n.Contains(c) ? confidence : wl.k[c]));
            }

        }
    }
}

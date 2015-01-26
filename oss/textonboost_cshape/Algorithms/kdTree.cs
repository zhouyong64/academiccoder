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

namespace Algorithms
{
    public class kdTree
    {
        // Euclidean Distance metric used
        // Splits on mean not median

        private class kdTreeNode
        {
            public bool terminal;

            // Terminal fields
            public int[] indices;

            // Non-terminal fields
            public int dim;
            public double val;
            public kdTreeNode left, right;

            // Terminal constructor
            public kdTreeNode(int[] indices)
            {
                terminal = true;
                this.indices = indices;
            }

            public void MakeNonTerminal(int dim, double val, kdTreeNode leftBranch, kdTreeNode rightBranch)
            {
                this.terminal = false;
                this.indices = null;
                this.dim = dim;
                this.val = val;
                this.left = leftBranch;
                this.right = rightBranch;
            }
        }

        private double[][] data;
        private int D;
        private kdTreeNode rootNode;

        // Build the kd-tree
        public kdTree(double[][] database, int pointsPerBin)
        {
            this.data = database;
            D = database[0].Length;

            // Initialise root node
            int[] indices = new int[database.Length];
            for (int i = 0; i < indices.Length; i++)
                indices[i] = i;
            rootNode = new kdTreeNode(indices);

            // Queue of nodes to process
            Queue<kdTreeNode> toProcess = new Queue<kdTreeNode>();
            toProcess.Enqueue(rootNode);

            // While queue is not empty
            while (toProcess.Count > 0)
            {
                kdTreeNode node = toProcess.Dequeue();
                int N = node.indices.Length;
                if (N > pointsPerBin)
                {
                    // Compute variances of each dimension for this bin
                    double[] vars = new double[D];
                    double[] x = new double[D], x2 = new double[D];
                    for(int i=0;i<N;i++)
                        for (int d = 0; d < D; d++)
                        {
                            x[d] += data[node.indices[i]][d];
                            x2[d] += data[node.indices[i]][d] * data[node.indices[i]][d];
                        }
                    for (int d = 0; d < D; d++)
                        vars[d] = x2[d] / N + x[d] * x[d] / (N * N);

                    // Find dimension of highest variance
                    int hDim = 0;
                    for (int d = 1; d < D; d++)
                        if (vars[d] > vars[hDim])
                            hDim = d;

                    // Check for all points equal - don't split any further
                    if (vars[hDim] == 0)
                        continue;

                    // Use mean of hDim to split at (should really be median, but saves sorting)
                    double split = x[hDim] / N;
                    List<int> leftIndices = new List<int>(), rightIndicies = new List<int>();
                    for (int i = 0; i < N; i++)
                        if (data[node.indices[i]][hDim] < split)
                            leftIndices.Add(node.indices[i]);
                        else
                            rightIndicies.Add(node.indices[i]);

                    // Make left and right tree nodes
                    kdTreeNode left = new kdTreeNode(leftIndices.ToArray());
                    kdTreeNode right = new kdTreeNode(rightIndicies.ToArray());

                    // Make the current node a non-terminal and recurse
                    node.MakeNonTerminal(hDim, split, left, right);
                    toProcess.Enqueue(left);
                    toProcess.Enqueue(right);
                }
            }
        }

        // Search the tree for a data-point
        public int NearestNeighbour(double[] point)
        {
            int bestIndex;
            double bestDistance;
            NearestNeighbour(point, rootNode, double.PositiveInfinity, out bestIndex, out bestDistance);
            return bestIndex;
        }

        private void NearestNeighbour(double[] point, kdTreeNode node, double currentBestDistance, out int bestIndex, out double bestDistance)
        {
            bestDistance = currentBestDistance;
            if (node.terminal)
            {
                // Linear search across data points in terminal node
                bestIndex = node.indices[0];
                bestDistance = EuclideanDistance(data[bestIndex], point, bestDistance);
                for (int i = 1; i < node.indices.Length; i++)
                {
                    int index = node.indices[i];
                    double dist = EuclideanDistance(data[index], point, bestDistance);
                    if (dist < bestDistance)
                    {
                        bestIndex = index;
                        bestDistance = dist;
                    }
                }
            }
            else
            {
                // Take branch according to split
                if (point[node.dim] < node.val)
                {
                    // Branch left
                    NearestNeighbour(point, node.left, bestDistance, out bestIndex, out bestDistance);

                    // Check if need to evaluate other branch
                    if (node.val - point[node.dim] <= bestDistance)
                    {
                        int checkIndex;
                        double checkDistance;
                        NearestNeighbour(point, node.right, bestDistance, out checkIndex, out checkDistance);
                        if (checkDistance < bestDistance)
                        {
                            bestIndex = checkIndex;
                            bestDistance = checkDistance;
                        }
                    }
                }
                else
                {
                    // Branch right
                    NearestNeighbour(point, node.right, bestDistance, out bestIndex, out bestDistance);

                    // Check if need to evaluate other branch
                    if (point[node.dim] - node.val <= bestDistance)
                    {
                        int checkIndex;
                        double checkDistance;
                        NearestNeighbour(point, node.left, bestDistance, out checkIndex, out checkDistance);
                        if (checkDistance < bestDistance)
                        {
                            bestIndex = checkIndex;
                            bestDistance = checkDistance;
                        }
                    }
                }
            }
        }

        private double EuclideanDistance(double[] p1, double[] p2, double minDistance)
        {
            minDistance = minDistance * minDistance; // Square minDistance so can be compared without square-rooting

            double euclideanDistance = 0.0;
            for (int d = 0; d < D; d++)
            {
                double diff = p1[d] - p2[d];
                euclideanDistance += diff * diff;
                if (euclideanDistance > minDistance) return double.PositiveInfinity; // Stop early if possible
            }
            return Math.Sqrt(euclideanDistance);
        }
    }
}

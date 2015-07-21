/**
 * Copyright (c) 2015, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */

package com.cloudera.examples

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.random.RandomRDDs.normalRDD
import org.apache.spark.mllib.stat.test.KolmogorovSmirnovTestResult

object KolmogorovSmirnovOLS {

  val sc = new SparkContext("local", "ks")

  def main(args: Array[String]): Unit = {

    val n = 1e6.toLong
    val x1 = normalRDD(sc, n, seed = 10L)
    val x2 = normalRDD(sc, n, seed = 5L)
    val X = x1.zip(x2)
    val y = X.map(x => 2.5 + 3.4 * x._1 + 1.5 * x._2)
    val data = y.zip(X).map { case (yi, (xi1, xi2)) => (yi, xi1, xi2)}

    val iters = 100
    val goodModelData = data.map(goodSpec)
    val goodModel = LinearRegressionWithSGD.train(goodModelData, iters)

    val badModelData = data.map(badSpec)
    val badModel = LinearRegressionWithSGD.train(badModelData, iters)

    val goodResiduals = goodModelData.map(p => p.label - goodModel.predict(p.features))
    val badResiduals = badModelData.map(p => p.label - badModel.predict(p.features))

    println(testResiduals(goodResiduals))
    println(testResiduals(badResiduals))
  }

  // This model specification captures all the regressors that underly
  // the true process
  def goodSpec(obs: (Double, Double, Double)): LabeledPoint = {
    val X = Array(1.0, obs._2, obs._3)
    val y = obs._1
    LabeledPoint(y, Vectors.dense(X)) //
  }

  def badSpec(obs: (Double, Double, Double)): LabeledPoint = {
    val X  = Array(1.0, obs._2, obs._2 * obs._1)
    val y = obs._1
    LabeledPoint(y, Vectors.dense(X))
  }

  // We'd like to compare our residuals against the standard normal distribution
  // But we'd like to use the 2-sample Kolmogorov-Smirnov test, so we'll draw a sample
  // from that distribution and compare
  def testResiduals(residuals: RDD[Double]): KolmogorovSmirnovTestResult = {
    val stdNormSample = normalRDD(sc, residuals.count())
    val mean = residuals.mean()
    val variance = residuals.map(x => (x - mean) * (x - mean)).mean()
    val standardizedResiduals = residuals.map(x => (x - mean) / math.sqrt(variance))
    // TODO: uncomment below once 2-sample included in MLlib
    //Statistics.kolmogorovSmirnovTest2(residuals, stdNormSample)
    Statistics.kolmogorovSmirnovTest(standardizedResiduals, "norm")
  }
}

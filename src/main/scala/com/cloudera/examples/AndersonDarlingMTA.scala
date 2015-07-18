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

import org.apache.commons.math3.distribution.ExponentialDistribution

import scala.util.Try

import org.joda.time.DateTime

import org.apache.spark.SparkContext
import org.apache.spark.mllib.stat.Statistics

object AndersonDarlingMTA {
  def main(a: Array[String]): Unit = {
    val sc = new SparkContext("local", "test")
    // TODO: replace with hdfs path in cluster
    val path = "/Users/josecambronero/Projects/gof/data/mta_may_2011"
    val mtaRaw = sc.textFile(path).sample(false, 0.3, 3L) // just a sample for now, since local
    val parsed = mtaRaw.map(x => Try(parseTrainEvent(x))).filter(_.isSuccess).map(_.get)
    // my train: 6, downtown, arrival times :)
    val train6Down= parsed.filter { x =>
      x.train == 6 && x.dir == 1 && x.eventType == 1
    }.sortBy(_.timestamp) // go ahead and sort now


    val stops = train6Down.map(_.stop).distinct
    val results = stops.map { stop =>
      val stopData = train6Down.filter(_.stop == stop).map(_.timestamp)
      // to calculate time span between arrivals we need to process elements sequentially
      // (to some extent). To take advantage of spark, we want to parallelize though
      // so the compromise is: processes partitions in parallel, and operate sequentially
      // within each partition. This means that we need to drop 1 span per partition (the first
      // one), and we drop any observations that are alone in a partition.
      val waitingTimes = stopData.mapPartitions { part =>
        val local = part.toArray
        local.zip(local.drop(1)).map { case (pt, t) =>
          (t.getMillis - pt.getMillis).toDouble / 6e4
        }.iterator
      }.filter(_ > 0)
      // We estimate the MLE for exponential (i.e. the mean of the data)
      val expDist = new ExponentialDistribution(waitingTimes.mean())
      // Just using KS as a place holder, this should be AD, since it is robust
      // against estimating parameters from the data that we're testing
      val result = Statistics.kolmogorovSmirnovTest(
        waitingTimes,
        (span: Double) => expDist.cumulativeProbability(span)
        )
      (stop, result)
    }
  }

  case class TrainEvent(
    timestamp: DateTime,
    train: Int,
    dir: Int,
    eventType: Int,
    route: String,
    stop: Int,
    track: String)


  def parseTrainEvent(line: String): TrainEvent = {
    val splitLine = line.replace("\"","").split(",")
    // skip service date, we have timestamp, which is all we vare about
    val train = splitLine(1).take(2).toInt // first 2 chars are train name
    val dir = splitLine(2).toInt //direction of train
    // make timestamp into ISO 8601 format for joda time
    val timestamp = new DateTime(splitLine(3).split("[ ]+").mkString("T"))
    val eventType = splitLine(4).toInt
    val route = splitLine(5)
    val stop = splitLine(6).toInt
    val track = splitLine(7)
    TrainEvent(timestamp, train, dir, eventType, route, stop, track)
  }
}

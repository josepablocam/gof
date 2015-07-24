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

import org.joda.time.{LocalTime, DateTime}
import org.joda.time.DateTimeConstants.{MONDAY, FRIDAY}

import org.apache.spark.rdd.RDD

import org.apache.spark.SparkContext
import org.apache.spark.mllib.stat.Statistics

object AndersonDarlingMTA {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local", "test")
    val path = "/Users/josecambronero/Projects/gof/data/mta_may_2011"
    val mtaRaw = sc.textFile(path)
    val parsed = mtaRaw.map(x => Try(parseTrainEvent(x))).filter(_.isSuccess).map(_.get)

    // get: weekday, between 7-8pm, excluding May 30th, 2011
    val rushHourData = parsed.filter { event =>
      val dateTime = event.timestamp
      val weekDay = isWeekDay(dateTime)
      val minTol = 1
      val rushHour = inTimeSlot(dateTime, new LocalTime(5, 0, 0), new LocalTime(8, 0, 0), minTol)
      // Memorial Day
      val holiday = new DateTime(2011, 5, 30, 0, 0, 0)
      val notHoliday = !dateTime.withTimeAtStartOfDay.isEqual(holiday.withTimeAtStartOfDay)
      weekDay && rushHour && notHoliday
    }

    // We'll focus on the 1 train, as it's schedule is regular, ie. there are no special
    // stops depending on time of day, and it runs "local", not skipping stations
    // http://web.mta.info/nyct/service/pdf/t1cur.pdf
    // We sort since we want data points from the same day to be close together
    val train1Down = rushHourData.filter (x =>
      x.train == 1 && x.dir == 1 && x.eventType == 1 && x.track.last == '1'
    ).sortBy(x => x.timestamp.getMillis)

    // Cache since we will be looking up against train1Down for each stop separately
    train1Down.cache()

    val stops = train1Down.map(_.stop).distinct().collect()
    val results = stops.map(stop => Try(testExpDistPerStop(train1Down, stop)))
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
    val route = splitLine(5).trim
    val stop = splitLine(6).toInt
    val track = splitLine(7).trim
    TrainEvent(timestamp, train, dir, eventType, route, stop, track)
  }

  def isWeekDay(dt: DateTime): Boolean = dt.getDayOfWeek >= MONDAY && dt.getDayOfWeek <= FRIDAY

  def inTimeSlot(
    dt: DateTime,
    start: LocalTime,
    end: LocalTime,
    tolMinutes: Int): Boolean = {
    require(end.compareTo(end) >= 0, s"$end is not >= $start")
    val tolerantStart = start.minusMinutes(tolMinutes)
    val tolerantEnd = end.plusMinutes(tolMinutes)
    val timeOfDay = dt.toLocalTime
    timeOfDay.compareTo(tolerantStart) >= 0 || timeOfDay.compareTo(tolerantEnd) <= 0
  }

  // to calculate time span between arrivals we need to process elements sequentially
  // (to some extent). To take advantage of spark, we want to parallelize though
  // so the compromise is: processes partitions in parallel, and operate sequentially
  // within each partition. This means we cannot calculate the span to the first observation
  // in a specific day in a partition, nor can we do so in the case of there
  // being only 1 day
  def getWaitingTimes(data: RDD[TrainEvent], stop: Int): RDD[Double] = {
    val stopData = data.filter(_.stop == stop).map(_.timestamp)
    val waitingTimes = stopData.mapPartitions { part =>
      val local = part.toArray
      // comparisons should only take place between arrival times in the same day
      local.groupBy(_.withTimeAtStartOfDay).flatMap { case (date, intraDayTimes) =>
        // sort times within a day, Scala's groupBy makes no guarantees on ordering
        val sortedMillis = intraDayTimes.map(_.getMillis).sortBy(x => x)
        sortedMillis.zip(sortedMillis.drop(1)).map { case (pt, t) =>
          (t - pt).toDouble / 6e4
        }
      }.filter(_ > 0).iterator
    }
    waitingTimes
  }

  def testExpDistPerStop(data: RDD[TrainEvent], stop: Int) = {
    val waitingTimes = getWaitingTimes(data, stop)
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

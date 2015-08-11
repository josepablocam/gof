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

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.random.{MersenneTwister, RandomGenerator}

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.test.AndersonDarlingTestResult
import org.apache.spark.SparkContext

import org.joda.time.{LocalDate, DateTime, LocalTime}
import org.joda.time.DateTimeConstants.{MONDAY, FRIDAY}

import scala.util.{Try, Success, Failure}

object AndersonDarlingAirlineDelays {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local", "ad")
    // Source: http://stat-computing.org/dataexpo/2009/the-data.html
    val path = getClass.getClassLoader.getResource("2008.csv").getFile
    val airLineRaw = sc.textFile(path)
    // wrap parser in Try to catch any exceptions, then get successes and extract
    val parsed = airLineRaw.map(x => Try(parseAirlineObs(x))).filter(_.isSuccess).map(_.get)

    // let's focus on SFO, during the work week, in the morning, 9:00am - 12pm
    val targetAirport = "SFO"
    val sfoDelayData = parsed.filter { obs =>
      val isSFO = obs.origin == targetAirport
      val isWorkWeek = isWeekDay(obs.timestamp)
      val isMorning = inTimeSlot(obs.timestamp, new LocalTime(9, 0, 0), new LocalTime(12, 0, 0), 1)
      val isDelayed = obs.departureDelay > 0
      val isNotFederalHoliday = !isFederalHoliday(obs.timestamp)
      isSFO && isWorkWeek && isMorning && isNotFederalHoliday && isDelayed
    }

    // We'll perform our analysis on the top 5 carriers with the most data points
    val topCarriers = sfoDelayData.groupBy(_.carrier).map(x => (x._2.size, x._1)).top(5).map(_._2)
    val dataForTop = sfoDelayData.filter(x => topCarriers.contains(x.carrier))

    // Cache since we will be looking up against dataForTop for each carrier separately
    dataForTop.cache()

    val rand = new MersenneTwister(10L)
    val logNormResults = topCarriers.map { carrier =>
      (carrier, Try(testLogNormDist(dataForTop, carrier, rand)))
    }
    logNormResults.foreach { case (carrier, result) =>
      println(s"--->$carrier")
     result match {
       case Success(ks) => println(ks)
       case Failure(_) => println("Failed to run")
     }
    }
  }

  case class AirlineObs(
    timestamp: DateTime, // year(1) + month(2) + day of month(3) + scheduled departure time ((6)
    carrier: String, // UniqueCarrier(9)
    tailNum: String, // plane tail number, unique identifier for plane (11)
    departureDelay: Int, // delays in minutes (16)
    origin: String, // airport code for origin (17)
    cancelled: Boolean // was flight canceled (22)
    )

  def parseAirlineObs(line: String): AirlineObs = {
    val splitLine = line.split(",")
    val Array(year, month, day) = splitLine.take(3).map(_.toInt)
    // pad in case formatting error
    val (strHr, strMin) = ("00" + splitLine(5)).takeRight(4).splitAt(2)
    val timestamp = new DateTime(year, month, day, strHr.toInt, strMin.toInt)
    val carrier = splitLine(8)
    val tailNum = splitLine(10)
    val departureDelay = splitLine(15).toInt
    val origin = splitLine(16)
    val cancelled = splitLine(21) == "1"
    AirlineObs(timestamp, carrier, tailNum, departureDelay, origin, cancelled)
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

  // month, day format
  val federalHolidays08 = List((1, 1), (1, 21), (2, 18), (4, 26), (7, 4), (9, 1), (10, 13),
    (11, 11), (11, 27), (12, 25)).map(x => new LocalDate(2008, x._1, x._2))

  def isFederalHoliday(dt: DateTime): Boolean = {
    federalHolidays08.contains(dt.toLocalDate)
  }

  // Following inspiration from
  // http://www.researchgate.net/publication/273946164_Modeling_Flight_Departure_Delay_Distributions
  def testLogNormDist(
    data: RDD[AirlineObs],
    carrier: String,
    rand: RandomGenerator): AndersonDarlingTestResult = {
    val delays = data.filter(_.carrier == carrier).map(_.departureDelay.toDouble)
    val hasTies = delays.distinct().count < delays.count
    val jittered = if (hasTies) delays.map(_ + rand.nextDouble / 100) else delays
    val jitteredLogged = jittered.map(math.log)
    // MLE estimates
    val location = jitteredLogged.mean()
    val scale = jitteredLogged.stdev()
    // We've logged transformed our data, so testing for normal here is equivalent
    // to testing the original series for lognormal distribution
    Statistics.andersonDarlingTest(jitteredLogged, "norm", location, scale)
  }
}

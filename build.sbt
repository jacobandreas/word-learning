organization := "edu.berkeley.cs.nlp"

name := "word_learning"

version := "0.1-SNAPSHOT"

scalaVersion := "2.11.5"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze-macros" % "0.11-M0",
  "org.scalanlp" %% "breeze" % "0.11-M0",
  "org.scalanlp" %% "breeze-config" % "0.9.1",
  "org.slf4j" % "slf4j-log4j12" % "1.7.12"
)

javaOptions ++= Seq("-Xmx4g", "-Xrunhprof:cpu=samples,depth=12")

resolvers ++= Seq(
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)


testOptions in Test += Tests.Argument("-oDF")

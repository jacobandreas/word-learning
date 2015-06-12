package word_learning

import java.io.{BufferedReader, FileReader, File}

import breeze.config.CommandLineParser
import breeze.linalg._
import breeze.numerics.exp
import breeze.optimize._
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.util.{HashIndex, Index}
import spire.syntax.cfor._

import scala.collection.mutable
import scala.io.Source

/**
 * Created by jda on 6/10/15.
 */

sealed trait Category
case object Noun extends Category
case object Adjective extends Category

case class Word(id: String, name: String, category: Category)

case class Object(id: String, embedding: DenseVector[Double])

case class Observation(objectId: Int, labelIds: IndexedSeq[Int])

case class Corpus(vocab: Index[Word],
                  objects: Index[Object],
                  observations: IndexedSeq[Observation]) {
  lazy val modelParams = {
    val objectParams = IndexedSeq.fill(vocab.size)(DenseVector.zeros[Double](objects.head.embedding.length))
    val priorParams = IndexedSeq.fill(1)(DenseVector.zeros[Double](0))
    Params(objectParams, priorParams)
  }
}

case class Params(objectParams: IndexedSeq[DenseVector[Double]],
                  priorParams: IndexedSeq[DenseVector[Double]]) {

  def +=(other: Params): Unit = {
    objectParams zip other.objectParams foreach { case (t, o) => t += o }
    priorParams zip other.priorParams foreach { case (t, o) => t += o }
  }

  def objectZero = DenseVector.zeros[Double](objectParams.head.length)
  def priorZero = DenseVector.zeros[Double](priorParams.head.length)

  def zero = Params(IndexedSeq.fill(objectParams.length)(objectZero),
                    IndexedSeq.fill(priorParams.length)(priorZero))

  def unpack(packedParams: DenseVector[Double]) = {
    val objectLen = objectParams.head.length
    val priorLen = priorParams.head.length

    val op = (0 until objectParams.length) map { i => packedParams(i * objectLen until (i+1) * objectLen) }
    val lastO = objectParams.length * objectLen
    val pp = (0 until priorParams.length) map { i => packedParams(lastO + i * priorLen until lastO + (i+1) * priorLen) }

    Params(op, pp)
  }

  def pack: DenseVector[Double] = DenseVector.vertcat((objectParams ++ priorParams): _*)
}

object Corpus {
  def load(dataDir: String): Corpus = {
    val attNameFile = new File(dataDir, "attnames.txt")
    val imageFile = new File(dataDir, "images.txt")
    val attFile = new File(dataDir, "atts.csv")
    val vecFile = new File(dataDir, "vectors.csv")

    val idxToAdj = mutable.Map[Int,Word]()
    val nameToNoun = mutable.Map[String,Word]()
    val nameToObj = mutable.Map[String,Object]()

    val wordIndex = new HashIndex[Word]()
    val objectIndex = new HashIndex[Object]()
    val used = Counter[Int,Int]()

    // load adjectives
    val attNameSrc = Source.fromFile(attNameFile)
    attNameSrc.getLines().zipWithIndex.foreach { case (attStr, i) =>
      val attId = s"a$i"
      val att = Word(attId, attStr, Adjective)
//      wordIndex.index(att)
      idxToAdj.put(i, att)
    }
    attNameSrc.close()

    // load nouns
    val nameSrc = Source.fromFile(imageFile)
    nameSrc.getLines().zipWithIndex.foreach { case (img, i) =>
      val synset = img.split("_")(0)
      val word = Word(synset, synset, Noun)
      nameToNoun.put(img, word)
    }
    nameSrc.close()

    // load objects
    val nameSrc2 = Source.fromFile(imageFile)
    val objectSrc = Source.fromFile(vecFile)
    for {
      (name, comps) <- nameSrc2.getLines() zip objectSrc.getLines()
      if comps.nonEmpty
    } {
      val vec = DenseVector(comps.split(",").map(_.toDouble))
      val obj = Object(name, vec)
      nameToObj.put(name, obj)
    }
    nameSrc2.close()
    objectSrc.close()

    // load observations
    val nameSrc3 = Source.fromFile(imageFile)
    val attSrc = Source.fromFile(attFile)
    val obs = for {
      (name, attStr) <- nameSrc3.getLines() zip attSrc.getLines() take 2000
      if nameToObj contains name
    } yield {
      val atts = attStr.split(",").zipWithIndex.filter(_._1 == "1").map(_._2)
      val adjs = atts map idxToAdj
      val noun = nameToNoun(name)
      val iObject = objectIndex.index(nameToObj(name))
      val iWords = (adjs :+ noun) map wordIndex.index
      Observation(iObject, iWords)
    }
    val obsSeq = obs.toIndexedSeq
    nameSrc3.close()
    attSrc.close()

    Corpus(wordIndex, objectIndex, obsSeq)
  }
}

case class Config(
  dataDir: String
)

object Main {
  def main(args: Array[String]): Unit = {
    val config = CommandLineParser.readIn[Config](args)
    val corpus = Corpus.load(config.dataDir)
    val objective = new Posterior(corpus)
    val init = DenseVector.zeros[Double](corpus.modelParams.pack.length)
    println(init.length + " features")
//    GradientTester.test[Int,DenseVector[Double]](objective, init)
    val optParamsPacked = minimize(objective, init, L1Regularization(0.1))
    val optParams = corpus.modelParams.unpack(optParamsPacked)
    optParams.objectParams.zipWithIndex.filter { case (d, i) => corpus.vocab.get(i).category == Noun }.map(_._1).foreach { p =>
      println(norm(p) + " " + sum(p) + " " + p.valuesIterator.toIndexedSeq.count(_.abs < 1e-5))
      println(p.valuesIterator.toIndexedSeq.count(p => 0 < p && p < 1e-5))
      println(p.valuesIterator.toIndexedSeq.count(p => -1e-5 < p && p < 0))
    }
    println("===")
    optParams.objectParams.zipWithIndex.filter { case (d, i) => corpus.vocab.get(i).category == Adjective }.map(_._1).foreach { p =>
      println(norm(p) + " " + sum(p) + " " + p.valuesIterator.toIndexedSeq.count(_.abs < 1e-5))
      println(p.valuesIterator.toIndexedSeq.count(p => 0 <= p && p < 1e-5))
      println(p.valuesIterator.toIndexedSeq.count(p => -1e-5 < p && p <= 0))
    }
  }

}

class Posterior(corpus: Corpus) extends DiffFunction[DenseVector[Double]] {

  def pObjects(params: Params): (Double, Params) = {
    var score = 0d
    val grad = params.zero

    corpus.observations.foreach { obs =>
      val aggWeights = obs.labelIds map params.objectParams reduce (_ + _)
      val objectScores = corpus.objects.map { o => aggWeights dot o.embedding }.toIndexedSeq
      val datumGrad = params.objectZero

      score += objectScores(obs.objectId)
      axpy(1d, corpus.objects.get(obs.objectId).embedding, datumGrad)

      val denom = softmax(objectScores)
      score -= denom
      cforRange (0 until corpus.objects.size) { oid =>
        val oprob = exp(objectScores(oid) - denom)
        axpy(-oprob, corpus.objects.get(oid).embedding, datumGrad)
      }

      obs.labelIds map grad.objectParams foreach (_ += datumGrad)
    }

    (score, grad)
  }

  def pObjectParams(params: Params): (Double, Params) = {
    return (0d, params.zero)
  }

  def pPriorParams(params: Params): (Double, Params) = {
    return (0d, params.zero)
  }

  override def calculate(packedParams: DenseVector[Double]): (Double, DenseVector[Double]) = {
    val params = corpus.modelParams.unpack(packedParams)

    val (objectScore, objectGrad) = pObjects(params)
    val (objectParamsScore, objectParamsGrad) = pObjectParams(params)
    val (priorParamsScore, priorParamsGrad) = pPriorParams(params)

    var score = 0d
    val grad = params.zero

    score += objectScore
    score += objectParamsScore
    score += priorParamsScore

    grad += objectGrad
    grad += objectParamsGrad
    grad += priorParamsGrad

//    println(grad.pack)
//    System.exit(1)

    (-score, -grad.pack)
  }
}

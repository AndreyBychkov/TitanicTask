import com.beust.klaxon.Klaxon
import java.io.File
import kotlin.math.exp
import kotlin.math.roundToInt

class LogRegressionModel(val coefficients: LogRegressionCoefficients) {

    private val coefficientsList: List<Float> = coefficients.asList()

    fun predictProbability(x: Collection<Float>): Float {
        val xWithIntercept = (x + 1.0) as List<Float>
        assert(xWithIntercept.size == coefficientsList.size)

        val expParam = scalarComposition(xWithIntercept, coefficientsList)
        val probability = 1 / (1 + exp(-expParam))

        return probability
    }

    fun predictOutcome(x: Collection<Float>) =
        predictProbability(x).roundToInt()

    private fun scalarComposition(first: Collection<Float>, second: Collection<Float>) =
        first.zip(second).map { it.first * it.second }.sum()

    companion object {
        fun readFromFile(file: File): LogRegressionModel {
            val coefficients = Klaxon().parse<LogRegressionCoefficients>(file)
            if (coefficients != null) {
                return LogRegressionModel(coefficients)
            } else {
                error("Can not parse Logistic Regression coefficients from file")
            }
        }

        fun readFromFile(filename: String): LogRegressionModel {
            val file = File(filename)
            return readFromFile(file)
        }
    }
}
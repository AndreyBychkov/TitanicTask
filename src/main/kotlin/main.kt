import krangl.*
import org.apache.commons.csv.CSVFormat
import java.io.File
import kotlin.math.roundToInt

fun <T> Array<T?>.replaceNull(replacement: T): Collection<T> =
    map { it ?: replacement }


fun encodeCategoricalColumns(data: DataFrame, encoders: List<LabelEncoder>): DataFrame {
    var encodedData = data
    encoders.forEach { encoder ->
        val colName = encoder.name
        encodedData = encodedData.addColumn(colName) { encoder.encode(it[colName].asStrings().filterNotNull()) }
    }
    return encodedData
}

fun inputeData(data: DataFrame): DataFrame {
    var imputedData = data

    imputedData.cols.forEach { col ->
        if (col is DoubleCol) {
            val mean = col.mean(removeNA = true)
            imputedData = imputedData.addColumn(col.name) { it[col.name].asDoubles().replaceNull(mean) }
        }

        if (col is IntCol) {
            val mode = col.mean(removeNA = true)!!.roundToInt()
            imputedData = imputedData.addColumn(col.name) { it[col.name].asInts().replaceNull(mode) }
        }
    }
    return imputedData
}

fun makePredictions(model: LogRegressionModel, data: DataFrame): List<Int> =
    data.rows.map { model.predictOutcome(it.values as Collection<Float>) }


fun main() {

    val encodersDir = File("data/model/encoders")
    val encoders = encodersDir.listFiles().map { LabelEncoder.readFromFile(it) }

    val modelCoefsDir = File("data/model/model_coefs.json")
    val model = LogRegressionModel.readFromFile(modelCoefsDir)

    var data = DataFrame.readCSV(
        "data/test.csv",
        CSVFormat.DEFAULT.withNullString("").withHeader()
    )

    var submission = data.select(listOf("PassengerId"))

    data = data.remove(listOf("PassengerId", "Name", "Ticket", "Cabin"))
    data = encodeCategoricalColumns(data, encoders)
    data = inputeData(data)

    submission = submission.addColumn("Survived") { makePredictions(model, data) }

    submission.writeCSV(File("data/submission.csv"))
}
import com.beust.klaxon.Klaxon
import java.io.File

class LabelEncoder(
    val name: String = "",
    val classes: List<String> = listOf()
) {

    private val classesMap = classes.mapIndexed { index, s -> s to index }.toMap()

    fun encode(x: String) =
        classesMap[x] ?: error("Cannot encode variable: class is not recognized")


    fun encode(x: Collection<String>) =
        x.map { encode(it) }

    companion object {
        fun readFromFile(filename: String): LabelEncoder {
            val file = File(filename)
            return readFromFile(file)
        }

        fun readFromFile(file: File): LabelEncoder {
            val name = file.nameWithoutExtension
            val fileData = Klaxon().parseArray<String>(file)

            if (fileData != null) {
                return LabelEncoder(name, fileData)
            } else {
                error("Can't read file data for LabelEncoder")
            }
        }
    }
}
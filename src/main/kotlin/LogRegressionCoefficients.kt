data class LogRegressionCoefficients(
    val Pclass: Float,
    val Sex: Float,
    val Age: Float,
    val SibSp: Float,
    val Parch: Float,
    val Fare: Float,
    val Embarked: Float,
    val Intercept: Float
) {

    fun asList(): List<Float>
        = listOf(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Intercept)

}
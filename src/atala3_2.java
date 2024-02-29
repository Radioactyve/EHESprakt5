import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

public class atala3_2 { //NaiveBayesPrediction
    public static void main(String[] args) throws Exception {
        // Cargar el modelo Naive Bayes previamente construido
        Classifier model = (Classifier) weka.core.SerializationHelper.read(args[0]);

        // Cargar el conjunto de datos de prueba
        DataSource source = new DataSource(args[1]);
        Instances testData = source.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1); // Establecer el índice de la clase

        // Obtener los atributos del modelo
        Instances modelHeader = new Instances(testData, 0); // Creamos una nueva instancia de Instances
        modelHeader.setClassIndex(testData.classIndex()); // Establecemos el índice de la clase según los datos de prueba
        for (int i = 0; i < testData.numAttributes(); i++) {
            Attribute attr = testData.attribute(i);
            if (attr != null && !attr.equals(testData.classAttribute())) {
                modelHeader.insertAttributeAt(attr, modelHeader.numAttributes());
            }
        }

        // Establecer el formato de los datos de prueba según el modelo
        testData = Instances.mergeInstances(testData, modelHeader);

        // Realizar predicciones en el conjunto de datos de prueba
        Evaluation evaluation = new Evaluation(testData);
        evaluation.evaluateModel(model, testData);

        // Imprimir los resultados de la evaluación
        System.out.println(evaluation.toSummaryString());
    }
}

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;

public class atala3 {
    public static void main(String[] args) throws Exception {
        // Cargar el modelo Naive Bayes previamente construido
        Classifier model = (Classifier) weka.core.SerializationHelper.read(args[0]);

        // Cargar el conjunto de datos de prueba
        DataSource source = new DataSource(args[1]);
        Instances testData = source.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1); // Establecer el índice de la clase

        // Aplicar el mismo filtro de selección de atributos al conjunto de datos de prueba
        AttributeSelection attributeSelectionFilter = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();
        attributeSelectionFilter.setEvaluator(eval);
        attributeSelectionFilter.setSearch(search);
        attributeSelectionFilter.setInputFormat(testData);
        Instances filteredTestData = Filter.useFilter(testData, attributeSelectionFilter);

        // Realizar predicciones en el conjunto de datos de prueba
        Evaluation evaluation = new Evaluation(filteredTestData);
        evaluation.evaluateModel(model, filteredTestData);

        // Imprimir los resultados de la evaluación
        System.out.println(evaluation.toSummaryString());
    }
}

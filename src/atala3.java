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
        // ------------------[INPUT]---------------------
        // Model
        String modelInput = args[0];
        Classifier model = (Classifier) weka.core.SerializationHelper.read(modelInput);
        // Prueba datuak
        String testInput = args[1];
        DataSource source = new DataSource(testInput);
        Instances testData = source.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        // --------------------[FILTER]---------------------
        AttributeSelection attributeSelectionFilter = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();
        attributeSelectionFilter.setEvaluator(eval);
        attributeSelectionFilter.setSearch(search);
        attributeSelectionFilter.setInputFormat(testData);
        Instances filteredTestData = Filter.useFilter(testData, attributeSelectionFilter);

        // --------------------[EVALUATION]-------------------
        Evaluation evaluation = new Evaluation(filteredTestData);
        evaluation.evaluateModel(model, filteredTestData);
        System.out.println(evaluation.toSummaryString());
    }
}

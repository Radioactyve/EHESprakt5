import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class atala4 {
    public static void main(String[] args) throws Exception {
        // -------------[INPUT]-----------
        // Train
        String trainInput = args[0];
        DataSource trainSource = new DataSource(trainInput);
        Instances trainData = trainSource.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1); // Establecer el índice de la clase
        // Test
        String testInput = args[1];
        DataSource testSource = new DataSource(testInput);
        Instances testData = testSource.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1); // Establecer el índice de la clase

        // ----------------------[NAIVE BAYES]-------------------------
        NaiveBayes baseClassifier = new NaiveBayes();

        // -----------------[EVALUATOR + SEARCHER]----------------------
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();
        // search.setSearchTermination(-1); //ez da guztiz beharrezkoa

        // -------------------[ATTRIBUTE SELECTED CLASSIFIER]-------------------------
        AttributeSelectedClassifier metaClassifier = new AttributeSelectedClassifier();
        metaClassifier.setClassifier(baseClassifier);
        metaClassifier.setEvaluator(eval);
        metaClassifier.setSearch(search);
        metaClassifier.buildClassifier(trainData);

        // Hautatutako atributuen indizeak
        int[] selectedAttributes = search.search(eval, trainData);

        // Ez hautatukoak kendu testData-tik
        Remove filter = new Remove();
        filter.setAttributeIndicesArray(selectedAttributes);
        filter.setInvertSelection(true);
        filter.setInputFormat(testData);
        Instances filteredTestData = Filter.useFilter(testData, filter);

        // ----------------------------[IRAGARPENAK]-------------------------
        double[] predictions = metaClassifier.distributionForInstance(filteredTestData.firstInstance());
        System.out.println("IRAGARPENAK:");
        for (int i = 0; i < predictions.length; i++) {
            System.out.println("Instantzia ID " + i + ": " + predictions[i]);
        }
    }
}

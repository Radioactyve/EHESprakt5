import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;

public class atala4 {
    public static void main(String[] args) throws Exception {
        // -------------[INPUT]-----------
        // Train
        String trainInput = args[0];
        DataSource trainSource = new DataSource(trainInput);
        Instances trainData = trainSource.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1);
        // Test
        String testInput = args[1];
        DataSource testSource = new DataSource(testInput);
        Instances testData = testSource.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        // ----------------------[NAIVE BAYES]-------------------------
        NaiveBayes baseClassifier = new NaiveBayes();

        // -----------------[EVALUATOR + SEARCHER]----------------------
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        // -------------------[ATTRIBUTE SELECTED CLASSIFIER]-------------------------
        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        classifier.setClassifier(baseClassifier);
        classifier.setEvaluator(eval);
        classifier.setSearch(search);
        classifier.buildClassifier(trainData);

        // Hautatutako atributuen indizeak
        int[] selectedAttributes = search.search(eval, trainData);

        // Ez hautatukoak kendu testData-tik
        Remove filter = new Remove();
        filter.setAttributeIndicesArray(selectedAttributes);
        filter.setInvertSelection(true);
        filter.setInputFormat(testData);
        Instances filteredTestData = Filter.useFilter(testData, filter);

        // ----------------------------[IRAGARPENAK]-------------------------
        Instances predictions = new Instances(testData);
        for (int i = 0; i < filteredTestData.numInstances(); i++) {
            double predictedClass = classifier.classifyInstance(filteredTestData.instance(i));
            predictions.instance(i).setClassValue(predictedClass);
        }
        // Emaitzak gorde
        FileWriter fw = new FileWriter(args[2]);
        fw.write(predictions.toString());
        fw.close();
    }
}

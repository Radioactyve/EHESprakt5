import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class atala3 {
    public static void main(String[] args) throws Exception {
        //------------------------[INPUT]-------------------------
        // Model
        String modelFile = args[0];
        NaiveBayes classifier = (NaiveBayes) weka.core.SerializationHelper.read(modelFile);
        // Test
        String testFile = args[1];
        Instances testData = new DataSource(testFile).getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);


        // ----------------------[FILTROA]----------------------
        // Indizeak lortu
        List<Integer> indicesToRemove = new ArrayList<>();
        for (int i = 0; i < classifier.getHeader().numAttributes(); i++) {
            Attribute attr = classifier.getHeader().attribute(i);
            int indexInTestData = testData.attribute(attr.name()).index();
            if (indexInTestData != -1) {
                indicesToRemove.add(indexInTestData);
            }
        }
        // Filtroa sortu
        Remove filter = new Remove();
        filter.setAttributeIndicesArray(indicesToRemove.stream().mapToInt(i -> i).toArray());
        filter.setInvertSelection(true);
        filter.setInputFormat(testData);
        // Filtroa aplikatu
        Instances filteredTestData = Filter.useFilter(testData, filter);


        // -----------------------[IRAGARPENAK]------------------------
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

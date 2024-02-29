import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

public class atala3_2 { //NaiveBayesPrediction
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

        // ----------------------[GET ATTRIBUTES]--------------------------------
        Instances modelHeader = new Instances(testData, 0); // instances motako balioak gordetzeko
        modelHeader.setClassIndex(testData.classIndex());

        for (int i = 0; i < testData.numAttributes(); i++) {
            Attribute attr = testData.attribute(i);
            if (attr != null && !attr.equals(testData.classAttribute())) {
                modelHeader.insertAttributeAt(attr, modelHeader.numAttributes());
            }
        }


        testData = Instances.mergeInstances(testData, modelHeader);
        Evaluation evaluation = new Evaluation(testData);
        evaluation.evaluateModel(model, testData);
        System.out.println(evaluation.toSummaryString());
    }
}

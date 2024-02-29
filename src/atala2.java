import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;

public class atala2 { //AttributeSelectionAndNaiveBayes
    public static void main(String[] args) throws Exception {
        // --------------[INPUT]---------------------
        String trainPath= args [0];
        DataSource source = new DataSource(trainPath);
        Instances trainData = source.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1);

        // --------------[ATRIBUTU HAUTAPENA]---------------------
        // -----------(CfsSubsetEval / BestFirst)------------
        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        classifier.setClassifier(new NaiveBayes());
        classifier.setEvaluator(new CfsSubsetEval());
        classifier.setSearch(new BestFirst());

        // ----------------[MODEL]---------------------
        classifier.buildClassifier(trainData);
        weka.core.SerializationHelper.write(args[1], classifier); //gorde

        // -----------------[INFO]---------------------
        System.out.println("Atributu kopurua hasieran: " + trainData.numAttributes());
        System.out.println("Atributu kopurua hautapena eta gero: " + classifier.measureNumAttributesSelected());
    }
}


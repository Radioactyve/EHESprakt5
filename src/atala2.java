import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class atala2 { //AttributeSelectionAndNaiveBayes
    public static void main(String[] args) throws Exception {
        // --------------[INPUT]---------------------
        String trainPath= args [0];
        DataSource source = new DataSource(trainPath);
        Instances trainData = source.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1);

        // --------------[ATRIBUTU HAUTAPENA]---------------------
        // (CfsSubsetEval / BestFirst)
        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(new CfsSubsetEval());
        selector.setSearch(new BestFirst());
        // (New Data)
        selector.setInputFormat(trainData);
        Instances newData = Filter.useFilter(trainData, selector);

        // ----------------[MODEL]---------------------
        NaiveBayes cls = new NaiveBayes();
        cls.buildClassifier(newData);
        weka.core.SerializationHelper.write(args[1], cls); //gorde

        // -----------------[INFO]---------------------
        System.out.println("Atributu kopurua hasieran: " + trainData.numAttributes());
        for (int i = 0; i < trainData.numAttributes(); i++) {
            System.out.println("Atributua " + (i+1) + ": " + trainData.attribute(i).name());
        }
        System.out.println("Atributu kopurua hautapena eta gero: " + newData.numAttributes());
        for (int i = 0; i < newData.numAttributes(); i++) {
            System.out.println("Atributua " + (i+1) + ": " + newData.attribute(i).name());
        }
    }
}


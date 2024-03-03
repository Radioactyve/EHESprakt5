import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Resample;

import java.io.File;

public class atala1 { //divide data set
    public static void main(String[] args) throws Exception {
        // -------------[INPUT]-----------
        String inputData = args[0];
        DataSource source = new DataSource(inputData);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // -------------[STRATIFIED BANAKETA]-----------
        // Settings
        Resample resampleFilter = new Resample();
        resampleFilter.setInputFormat(data);
        resampleFilter.setNoReplacement(true);
        resampleFilter.setSampleSizePercent(70.0);

        // Train
        Instances trainData = Filter.useFilter(data, resampleFilter);

        // Dev
        resampleFilter.setInputFormat(data);
        resampleFilter.setSampleSizePercent(70.0);
        resampleFilter.setInvertSelection(true);
        Instances devData = Filter.useFilter(data, resampleFilter);

        // ----------------[SAVE ARFF]-------------------
        // TRAIN
        ArffSaver trainSaver = new ArffSaver();
        trainSaver.setInstances(trainData);
        trainSaver.setFile(new File(args[1]));
        trainSaver.writeBatch();

        // DEV "klase balioak --> ?"
        for (int i=0; i<devData.size();i++){
            devData.instance(i).setMissing(data.numAttributes() - 1);
        }
        ArffSaver devReplacedSaver = new ArffSaver();
        devReplacedSaver.setInstances(devData);
        devReplacedSaver.setFile(new File(args[2]));
        devReplacedSaver.writeBatch();
    }
}


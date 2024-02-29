import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.File;

public class atala1 { //divide data set
    public static void main(String[] args) throws Exception {
        // -------------[INPUT]-----------
        String inputData = args[0];
        DataSource source = new DataSource(inputData);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // -------------[STRATIFIED BANAKETA]-----------
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
        filter.setInputFormat(data);
        filter.setNumFolds(10); // Dividir en 10 pliegues para obtener el 70% y 30%
        filter.setFold(1); // Mantener solo el primer pliegue para obtener el 70%
        Instances trainData = Filter.useFilter(data, filter);
        filter.setInvertSelection(true); // Mantener la parte que no se seleccionó (30%)
        Instances devData = Filter.useFilter(data, filter);

        // ----------------[SAVE ARFF]-------------------
        // TRAIN
        ArffSaver trainSaver = new ArffSaver();
        trainSaver.setInstances(trainData);
        trainSaver.setFile(new File(args[1]));
        trainSaver.writeBatch();
        // TEST "klase balioak --> ?"
        ReplaceMissingValues replace = new ReplaceMissingValues();
        replace.setInputFormat(devData);
        Instances devDataReplaced = Filter.useFilter(devData, replace);

        ArffSaver devReplacedSaver = new ArffSaver();
        devReplacedSaver.setInstances(devDataReplaced);
        devReplacedSaver.setFile(new File(args[2]));
        devReplacedSaver.writeBatch();
    }
}


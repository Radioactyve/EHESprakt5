import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;

public class atala4 {
    public static void main(String[] args) throws Exception {
        // Cargar conjunto de datos de entrenamiento
        DataSource source = new DataSource("train.arff");
        Instances trainData = source.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1); // Establecer el índice de la clase

        // Crear el clasificador Naive Bayes base
        NaiveBayes baseClassifier = new NaiveBayes();

        // Crear el filtro de selección de atributos CfsSubsetEval y BestFirst
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();
        search.setSearchTermination(-1); // Terminar la búsqueda de atributos cuando se alcanza el máximo

        // Crear el clasificador meta con selección de atributos
        AttributeSelectedClassifier metaClassifier = new AttributeSelectedClassifier();
        metaClassifier.setClassifier(baseClassifier);
        metaClassifier.setEvaluator(eval);
        metaClassifier.setSearch(search);

        // Crear el clasificador filtrado que maneja la inconsistencia causada por los filtros de selección de atributos
        FilteredClassifier filteredClassifier = new FilteredClassifier();
        filteredClassifier.setClassifier(metaClassifier);
        filteredClassifier.buildClassifier(trainData);

        // Obtener el evaluador y el buscador de atributos
        ASEvaluation attributeEvaluator = eval;
        ASSearch attributeSearch = search;

        // Obtener los atributos seleccionados
        int[] selectedAttributes = attributeSearch.search(attributeEvaluator, trainData);

        // Imprimir los atributos seleccionados
        System.out.println("Atributos seleccionados:");
        for (int attributeIndex : selectedAttributes) {
            System.out.println("- Atributo " + attributeIndex + ": " + trainData.attribute(attributeIndex).name());
        }
    }
}

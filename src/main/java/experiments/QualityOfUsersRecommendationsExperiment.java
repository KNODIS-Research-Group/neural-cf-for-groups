package main.java.experiments;

import cf4j.Kernel;
import cf4j.Processor;
import cf4j.model.matrixFactorization.Bmf;
import cf4j.model.matrixFactorization.FactorizationModel;
import cf4j.model.matrixFactorization.Pmf;
import cf4j.model.predictions.FactorizationPrediction;
import cf4j.qualityMeasures.MSD;
import cf4j.qualityMeasures.Precision;
import cf4j.qualityMeasures.Recall;
import cf4j.utils.Range;
import main.java.baselines.*;
import main.java.data.GroupsManager;
import main.java.gumf.CoGrmf;
import main.java.gumf.Grmf;
import main.java.gumf.GroupRecommender;
import main.java.misc.AggregationFunctions;
import main.java.qualityMeasures.GroupQualityMeasures;
import main.java.qualityMeasures.Ndcg;


public class QualityOfUsersRecommendationsExperiment {

//    final static String CF4J_FILE = "ML1M.cf4j";
//    final static String GROUPS_FILE = "GROUPS-ML1M.bin";
//
//    final static int NUM_FACTORS = 10;
//    final static int NUM_ITERS = 200;
//    final static double REGULARIZATION = 0.05;
//    final static double LEARNING_RATE = 0.01;
//
//    final static double THRESHOLD = 4;

    final static String CF4J_FILE = "ML10M.cf4j";
    final static String GROUPS_FILE = "GROUPS-ML10M.bin";

    final static int NUM_FACTORS = 10;
    final static int NUM_ITERS = 200;
    final static double REGULARIZATION = 0.05;
    final static double LEARNING_RATE = 0.01;

    final static double THRESHOLD = 4;

    final static String [] METHODS = {"CoGRMF", "PMF", "NMF", "BNMF", "BiasedMF"};
    final static int [] NUM_RECOMMENDATIONS = Range.ofIntegers(1,1,20);

    public static void main (String[] args) throws Exception {

        Kernel.getInstance().readKernel(CF4J_FILE);
        GroupsManager.read(GROUPS_FILE);

        double [] rmse = new double [METHODS.length];
        double [][] precision = new double [METHODS.length][NUM_RECOMMENDATIONS.length];
        double [][] recall = new double [METHODS.length][NUM_RECOMMENDATIONS.length];
        double [][] f1 = new double [METHODS.length][NUM_RECOMMENDATIONS.length];
        double [][] ndcg = new double [METHODS.length][NUM_RECOMMENDATIONS.length];


        for (int m = 0; m < METHODS.length; m++) {
            String method = METHODS[m];

            FactorizationModel fm = null;

            System.out.println("\nProcessing " + method + "...");

            // train user recommendation model
            if (method.equals("CoGRMF")) {

                CoGrmf coGrmf = new CoGrmf(NUM_FACTORS, NUM_ITERS, REGULARIZATION, LEARNING_RATE, AggregationFunctions.sum());
                coGrmf.train();
                fm = coGrmf;

            } else if (method.equals("PMF")) {

                Pmf pmf = new Pmf(NUM_FACTORS, NUM_ITERS, REGULARIZATION, LEARNING_RATE, false);
                pmf.train();
                fm = pmf;

            } else if (method.equals("BiasedMF")) {

                Pmf pmf = new Pmf(NUM_FACTORS, NUM_ITERS, REGULARIZATION, LEARNING_RATE, true);
                pmf.train();
                fm = pmf;

            } else if (method.equals("NMF")) {

                Nmf nmf = new Nmf(NUM_FACTORS, NUM_ITERS);
                nmf.train();
                fm = nmf;

            } else if (method.equals("BNMF")) {

                Bmf bnmf = new Bmf(NUM_FACTORS, NUM_ITERS, 0.8, 5);
                bnmf.train();
                fm = bnmf;

            }

            Processor.getInstance().testUsersProcess(new FactorizationPrediction(fm));


            // compute user based quality measures

            Processor.getInstance().testUsersProcess(new MSD());
            rmse[m] = Math.sqrt(Kernel.getInstance().getQualityMeasure("MSD"));

            for (int n = 0; n < NUM_RECOMMENDATIONS.length; n++) {
                int RECS = NUM_RECOMMENDATIONS[n];

                Processor.getInstance().testUsersProcess(new Precision(RECS, THRESHOLD));
                precision[m][n] = Kernel.getInstance().getQualityMeasure("Precision");

                Processor.getInstance().testUsersProcess(new Recall(RECS, THRESHOLD));
                recall[m][n] = Kernel.getInstance().getQualityMeasure("Recall");

                f1[m][n] = 2 * precision[m][n] * recall[m][n] / (precision[m][n] + recall[m][n]);

                Processor.getInstance().testUsersProcess(new Ndcg(RECS));
                ndcg[m][n] = Kernel.getInstance().getQualityMeasure("NDCG");
            }

            // print results
            printRmse(rmse);
            printPrecision(precision);
            printRecall(recall);
            printF1(f1);
            printNdcg(ndcg);
        }
    }

    private static void printRmse (double [] rmse) {
        System.out.println("\nMethod;RMSE");
        for (int m = 0; m < METHODS.length; m++) {
            System.out.println(METHODS[m] + ";" + rmse[m]);
        }
    }

    private static void printPrecision (double [][] precision) {
        printRecommendationQualityMeasure("Precision", precision);
    }

    private static void printRecall (double [][] recall) {
        printRecommendationQualityMeasure("Recall", recall);
    }

    private static void printF1 (double [][] f1) {
        printRecommendationQualityMeasure("F1", f1);
    }

    private static void printNdcg (double [][] ndcg) {
        printRecommendationQualityMeasure("nDCG", ndcg);
    }

    private static void printRecommendationQualityMeasure (String name, double [][] measure) {
        System.out.print("\n" + name);
        for (String method : METHODS) System.out.print(";" + method);
        System.out.println();

        for (int n = 0; n < NUM_RECOMMENDATIONS.length; n++) {
            System.out.print(NUM_RECOMMENDATIONS[n]);
            for (int m = 0; m < METHODS.length; m++) {
                System.out.print(";" + measure[m][n]);
            }
            System.out.println();
        }
    }
}

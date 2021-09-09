package main.java.experiments;

import cf4j.Kernel;
import cf4j.utils.Range;
import main.java.baselines.*;
import main.java.data.GroupsManager;
import main.java.gumf.CoGrmf;
import main.java.gumf.Grmf;
import main.java.gumf.GroupRecommender;
import main.java.misc.AggregationFunctions;
import main.java.qualityMeasures.GroupQualityMeasures;


public class QualityOfGroupsRecommendationsExperiment {

    // MovieLens 1M

//    final static String CF4J_FILE = "ML1M.cf4j";
//    final static String GROUPS_FILE = "GROUPS-ML1M.bin";
//
//    final static int MF_NUM_FACTORS = 10;
//    final static int MF_NUM_ITERS = 200;
//    final static double MF_REGULARIZATION = 0.05;
//    final static double MF_LEARNING_RATE = 0.01;
//
//    final static double SMGU_ALPHA = 0.2;
//    final static int SMGU_NUM_NEIGHBORS = 350;
//
//    final static double RWRM_RHO = 0.15;
//    final static int RWRM_NUM_ITERS = 25;
//
//    final static int CP_NUM_NEIGHBORS = 350;
//
//    final static int RANK_NUM_NEIGHBORS = 350;
//
//    final static double THRESHOLD = 4;


    // MovieLens 10M

//    final static String CF4J_FILE = "ML10M.cf4j";
//    final static String GROUPS_FILE = "GROUPS-ML10M.bin";
//
//    final static int MF_NUM_FACTORS = 10;
//    final static int MF_NUM_ITERS = 200;
//    final static double MF_REGULARIZATION = 0.05;
//    final static double MF_LEARNING_RATE = 0.01;
//
//    final static double SMGU_ALPHA = 0.2;
//    final static int SMGU_NUM_NEIGHBORS = 350;
//
//    final static double RWRM_RHO = 0.15;
//    final static int RWRM_NUM_ITERS = 25;
//
//    final static int CP_NUM_NEIGHBORS = 350;
//
//    final static int RANK_NUM_NEIGHBORS = 350;
//
//    final static double THRESHOLD = 4;


    // FilmTrust

//    final static String CF4J_FILE = "FT.cf4j";
//    final static String GROUPS_FILE = "GROUPS-FT.bin";
//
//    final static int MF_NUM_FACTORS = 10;
//    final static int MF_NUM_ITERS = 300;
//    final static double MF_REGULARIZATION = 0.095;
//    final static double MF_LEARNING_RATE = 0.035;
//
//    final static double SMGU_ALPHA = 0.5;
//    final static int SMGU_NUM_NEIGHBORS = 300;
//
//    final static double RWRM_RHO = 0.15;
//    final static int RWRM_NUM_ITERS = 25;
//
//    final static int CP_NUM_NEIGHBORS = 70;
//
//    final static int RANK_NUM_NEIGHBORS = 70;
//
//    final static double THRESHOLD = 3;


    // BookCrossing

//    final static String CF4J_FILE = "BX.cf4j";
//    final static String GROUPS_FILE = "GROUPS-BX.bin";
//
//    final static int MF_NUM_FACTORS = 10;
//    final static int MF_NUM_ITERS = 300;
//    final static double MF_REGULARIZATION = 0.095;
//    final static double MF_LEARNING_RATE = 0.035;
//
//    final static double SMGU_ALPHA = 0.15;
//    final static int SMGU_NUM_NEIGHBORS = 75;
//
//    final static double RWRM_RHO = 0.15;
//    final static int RWRM_NUM_ITERS = 25;
//
//    final static int CP_NUM_NEIGHBORS = 70;
//
//    final static int RANK_NUM_NEIGHBORS = 70;
//
//    final static double THRESHOLD = 8;


    // CiaoDVD

//    final static String CF4J_FILE = "CiaoDVD.cf4j";
//    final static String GROUPS_FILE = "GROUPS-CiaoDVD.bin";
//
//    final static int MF_NUM_FACTORS = 8;
//    final static int MF_NUM_ITERS = 200;
//    final static double MF_REGULARIZATION = 0.06;
//    final static double MF_LEARNING_RATE = 0.008;
//
//    final static double SMGU_ALPHA = 0.25;
//    final static int SMGU_NUM_NEIGHBORS = 200;
//
//    final static double RWRM_RHO = 0.18;
//    final static int RWRM_NUM_ITERS = 20;
//
//    final static int CP_NUM_NEIGHBORS = 200;
//
//    final static int RANK_NUM_NEIGHBORS = 200;
//
//    final static double THRESHOLD = 4;


    // CiaoDVD

    final static String CF4J_FILE = "Flixter.cf4j";
    final static String GROUPS_FILE = "GROUPS-Flixter.bin";

    final static int MF_NUM_FACTORS = 10;
    final static int MF_NUM_ITERS = 200;
    final static double MF_REGULARIZATION = 0.05;
    final static double MF_LEARNING_RATE = 0.01;

    final static double SMGU_ALPHA = 0.2;
    final static int SMGU_NUM_NEIGHBORS = 350;

    final static double RWRM_RHO = 0.15;
    final static int RWRM_NUM_ITERS = 25;

    final static int CP_NUM_NEIGHBORS = 350;

    final static int RANK_NUM_NEIGHBORS = 350;

    final static double THRESHOLD = 4;


//    final static String [] METHODS = {"CoGRMF", "GRMF", "MFGU", "SMGU", "RWR-M", "C&P", "RANK"};
    final static String [] METHODS = {"SMGU", "RWR-M", "C&P", "RANK"};
    final static int [] NUM_RECOMMENDATIONS = Range.ofIntegers(1,1,20);
    final static double [] PHIS = Range.ofDoubles(0.50,0.25,3);

    public static void main (String[] args) throws Exception {

        Kernel.getInstance().readKernel(CF4J_FILE);
        GroupsManager.read(GROUPS_FILE);

        double [] rmse = new double [METHODS.length];
        double [][][] precision = new double [PHIS.length][METHODS.length][NUM_RECOMMENDATIONS.length];
        double [][][] recall = new double [PHIS.length][METHODS.length][NUM_RECOMMENDATIONS.length];
        double [][][] f1 = new double [PHIS.length][METHODS.length][NUM_RECOMMENDATIONS.length];
        double [][] ndcg = new double [METHODS.length][NUM_RECOMMENDATIONS.length];


        for (int m = 0; m < METHODS.length; m++) {
            String method = METHODS[m];

            GroupRecommender gr = null;

            System.out.println("\nProcessing " + method + "...");

            // train group recommendation model
            if (method.equals("CoGRMF")) {

                CoGrmf coGrmf = new CoGrmf(MF_NUM_FACTORS, MF_NUM_ITERS, MF_REGULARIZATION, MF_LEARNING_RATE, AggregationFunctions.sum());
                coGrmf.train();
                gr = coGrmf;

            } else if (method.equals("GRMF")) {

                Grmf grmf = new Grmf(MF_NUM_FACTORS, MF_NUM_ITERS, MF_REGULARIZATION, MF_LEARNING_RATE, AggregationFunctions.sum());
                grmf.train();
                gr = grmf;

            } else if (method.equals("MFGU")) {

                Mfgu mfgu = new Mfgu(MF_NUM_FACTORS, MF_NUM_ITERS, MF_REGULARIZATION, MF_LEARNING_RATE);
                mfgu.train();
                gr = mfgu;

            } else if (method.equals("SMGU")) {

                Smgu smgu = new Smgu(SMGU_ALPHA, SMGU_NUM_NEIGHBORS);
                smgu.fit();
                gr = smgu;

            } else if (method.equals("RWR-M")) {

                Rwrm rwrm = new Rwrm(RWRM_RHO, RWRM_NUM_ITERS);
                rwrm.fit();
                gr = rwrm;

            } else if (method.equals("C&P")) {

                ClusterAndPredict cp = new ClusterAndPredict(CP_NUM_NEIGHBORS);
                cp.fit();
                gr = cp;

            } else if (method.equals("RANK")) {

                Rank rank = new Rank(RANK_NUM_NEIGHBORS);
                rank.fit();
                gr = rank;
            }


            // compute group based quality measures
            rmse[m] = GroupQualityMeasures.getNormalizedRmse(gr, AggregationFunctions.sum());

            for (int n = 0; n < NUM_RECOMMENDATIONS.length; n++) {
                int RECS = NUM_RECOMMENDATIONS[n];

                for (int p = 0; p < PHIS.length; p++) {
                    double phi = PHIS[p];

                    precision[p][m][n] = GroupQualityMeasures.getPrecision(gr, RECS, THRESHOLD, phi);
                    recall[p][m][n] = GroupQualityMeasures.getRecall(gr, RECS, THRESHOLD, phi);
                    f1[p][m][n] = 2 * precision[p][m][n] * recall[p][m][n] / (precision[p][m][n] + recall[p][m][n]);
                }

                ndcg[m][n] = GroupQualityMeasures.getNdcg(gr, RECS, AggregationFunctions.sum());
            }

            // print results
            printRmse(rmse);

            for (int p = 0; p < PHIS.length; p++) {
                double phi = PHIS[p];
                printPrecision(phi, precision[p]);
                printRecall(phi, recall[p]);
                printF1(phi, f1[p]);
            }

            printNdcg(ndcg);
        }
    }

    private static void printRmse (double [] rmse) {
        System.out.println("\nMethod;RMSE");
        for (int m = 0; m < METHODS.length; m++) {
            System.out.println(METHODS[m] + ";" + rmse[m]);
        }
    }

    private static void printPrecision (double phi, double [][] precision) {
        printRecommendationQualityMeasure("Precision@" + phi, precision);
    }

    private static void printRecall (double phi, double [][] recall) {
        printRecommendationQualityMeasure("Recall@" + phi, recall);
    }

    private static void printF1 (double phi, double [][] f1) {
        printRecommendationQualityMeasure("F1@" + phi, f1);
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

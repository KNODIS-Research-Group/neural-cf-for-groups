package main.java.experiments;

import cf4j.Kernel;
import cf4j.Processor;
import cf4j.model.predictions.FactorizationPrediction;
import cf4j.qualityMeasures.MAE;
import main.java.baselines.*;
import main.java.data.GroupsManager;
import main.java.gumf.CoGrmf;
import main.java.gumf.Grmf;
import main.java.misc.AggregationFunctions;
import main.java.qualityMeasures.GroupQualityMeasures;
import org.apache.commons.math3.ml.clustering.Cluster;

import java.util.Iterator;
import java.util.function.Function;


public class PreliminaryExperiment {

    final static String CF4J_FILE = "ML1M.cf4j";
    final static String GROUPS_FILE = "GROUPS-ML1M.bin";

    final static int NUM_FACTORS = 10;
    final static int NUM_ITERS = 200;
    final static double REGULARIZATION = 0.05;
    final static double LEARNING_RATE = 0.01;

    final static int NUM_RECOMMENDATIONS = 10;
    final static double THRESHOLD = 4;
    final static double PHI = 0.75;

    final static String [] merges = {"min", "max", "avg", "geo", "sum"};

    public static void main (String[] args) throws Exception {

        Kernel.getInstance().readKernel(CF4J_FILE);
        GroupsManager.read(GROUPS_FILE);

        // CoGRMF

        for (String merge : merges) {
            Function<Iterator<Double>, Double> mergeFunc = getMergeFunction(merge);

            CoGrmf coGrmf = new CoGrmf(NUM_FACTORS,NUM_ITERS,REGULARIZATION,LEARNING_RATE, mergeFunc);
            coGrmf.train();

            System.out.println("CoGRMF (" + merge + ") rmse = " + GroupQualityMeasures.getNormalizedRmse(coGrmf, mergeFunc));

            System.out.println("CoGRMF (" + merge + ") precision = " + GroupQualityMeasures.getPrecision(coGrmf, NUM_RECOMMENDATIONS, THRESHOLD, PHI));
            System.out.println("CoGRMF (" + merge + ") recall = " + GroupQualityMeasures.getRecall(coGrmf, NUM_RECOMMENDATIONS, THRESHOLD, PHI));

            for (String m : merges) {
                System.out.println("CoGRMF (" + merge + ") nDCG (" + m + ") = " + GroupQualityMeasures.getNdcg(coGrmf, NUM_RECOMMENDATIONS, getMergeFunction(m)));
            }

            Processor.getInstance().testUsersProcess(new FactorizationPrediction(coGrmf));
            Processor.getInstance().testUsersProcess(new MAE());

            System.out.println("CoGRMF (" + merge + ") user mae = " + Kernel.getInstance().getQualityMeasure("MAE"));
        }


        // GRMF

        for (String merge : merges) {
            Function<Iterator<Double>, Double> mergeFunc = getMergeFunction(merge);

            Grmf grmf = new Grmf(NUM_FACTORS, NUM_ITERS, REGULARIZATION, LEARNING_RATE, mergeFunc);
            grmf.train();

            System.out.println("GRMF (" + merge + ") rmse = " + GroupQualityMeasures.getNormalizedRmse(grmf, mergeFunc));

            System.out.println("GRMF (" + merge + ") precision = " + GroupQualityMeasures.getPrecision(grmf, NUM_RECOMMENDATIONS, THRESHOLD, PHI));
            System.out.println("GRMF (" + merge + ") recall = " + GroupQualityMeasures.getRecall(grmf, NUM_RECOMMENDATIONS, THRESHOLD, PHI));

            for (String m : merges) {
                System.out.println("GRMF (" + merge + ") nDCG (" + m + ") = " + GroupQualityMeasures.getNdcg(grmf, NUM_RECOMMENDATIONS, getMergeFunction(m)));
            }
        }


        // MFGU

        Mfgu mfgu = new Mfgu(NUM_FACTORS, NUM_ITERS, REGULARIZATION, LEARNING_RATE);
        mfgu.train();

        System.out.println("MFGU rmse = " + GroupQualityMeasures.getNormalizedRmse(mfgu, AggregationFunctions.min()));

        System.out.println("MFGU precision = " + GroupQualityMeasures.getPrecision(mfgu,NUM_RECOMMENDATIONS, THRESHOLD, PHI));
        System.out.println("MFGU recall = " + GroupQualityMeasures.getRecall(mfgu,NUM_RECOMMENDATIONS, THRESHOLD, PHI));

        for (String m : merges) {
            System.out.println("MFGU nDCG (" + m + ") = " + GroupQualityMeasures.getNdcg(mfgu, NUM_RECOMMENDATIONS, getMergeFunction(m)));
        }

        Processor.getInstance().testUsersProcess(new FactorizationPrediction(mfgu.getFactorizationModel()));
        Processor.getInstance().testUsersProcess(new MAE());

        System.out.println("MFGU user mae = " + Kernel.getInstance().getQualityMeasure("MAE"));


        // SMGU

        Smgu smgu = new Smgu(0.2, 350);
        smgu.fit();

        System.out.println("SMGU rmse = " + GroupQualityMeasures.getNormalizedRmse(smgu, AggregationFunctions.avg()));

        System.out.println("SMGU precision = " + GroupQualityMeasures.getPrecision(smgu,NUM_RECOMMENDATIONS, THRESHOLD, PHI));
        System.out.println("SMGU recall = " + GroupQualityMeasures.getRecall(smgu,NUM_RECOMMENDATIONS, THRESHOLD, PHI));

        for (String m : merges) {
            System.out.println("SMGU nDCG (" + m + ") = " + GroupQualityMeasures.getNdcg(smgu, NUM_RECOMMENDATIONS, getMergeFunction(m)));
        }


        // RWR-M

        Rwrm rwrm = new Rwrm(0.15, 25);
        rwrm.fit();

        System.out.println("RWR-M precision = " + GroupQualityMeasures.getPrecision(rwrm,NUM_RECOMMENDATIONS, THRESHOLD, PHI));
        System.out.println("RWR-M recall = " + GroupQualityMeasures.getRecall(rwrm,NUM_RECOMMENDATIONS, THRESHOLD, PHI));

        for (String m : merges) {
            System.out.println("RWR-M nDCG (" + m + ") = " + GroupQualityMeasures.getNdcg(rwrm, NUM_RECOMMENDATIONS, getMergeFunction(m)));
        }


        // C&P

        ClusterAndPredict cp = new ClusterAndPredict(350);
        cp.fit();

        System.out.println("C&P rmse = " + GroupQualityMeasures.getNormalizedRmse(cp, AggregationFunctions.avg()));

        System.out.println("C&P precision = " + GroupQualityMeasures.getPrecision(cp,NUM_RECOMMENDATIONS, THRESHOLD, PHI));
        System.out.println("C&P recall = " + GroupQualityMeasures.getRecall(cp,NUM_RECOMMENDATIONS, THRESHOLD, PHI));

        for (String m : merges) {
            System.out.println("C&P nDCG (" + m + ") = " + GroupQualityMeasures.getNdcg(cp, NUM_RECOMMENDATIONS, getMergeFunction(m)));
        }


        // Rank

        Rank rank = new Rank(350);
        rank.fit();

        System.out.println("RANK precision = " + GroupQualityMeasures.getPrecision(rank,NUM_RECOMMENDATIONS, THRESHOLD, PHI));
        System.out.println("RANK recall = " + GroupQualityMeasures.getRecall(rank,NUM_RECOMMENDATIONS, THRESHOLD, PHI));

        for (String m : merges) {
            System.out.println("RANK nDCG (" + m + ") = " + GroupQualityMeasures.getNdcg(rank, NUM_RECOMMENDATIONS, getMergeFunction(m)));
        }
    }

    private static Function<Iterator<Double>, Double> getMergeFunction (String name) {
        switch (name) {
            case "min": return AggregationFunctions.min();
            case "max": return AggregationFunctions.max();
            case "avg": return AggregationFunctions.avg();
            case "geo": return AggregationFunctions.geo();
            case "sum": return AggregationFunctions.sum();
            case "prod": return AggregationFunctions.prod();
            default: return null;
        }
    }
}

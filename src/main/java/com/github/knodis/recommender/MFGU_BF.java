package com.github.knodis.recommender;

import com.github.knodis.data.GroupManager;
import com.github.knodis.data.Sample;
import com.github.knodis.etc.Config;
import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BiasedMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;
import es.upm.etsisi.cf4j.util.Maths;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.ejml.simple.SimpleMatrix;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class MFGU_BF {

    public static void main (String[] args) throws Exception {
        DataModel datamodel = null;

        if (Config.DB_NAME.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        }

        BiasedMF biasedMF = new BiasedMF(datamodel, 6,50, 0.05, 0.01, Config.RANDOM_SEED);
        biasedMF.fit();

        int groupSize = 4;

        GroupManager groupManager = new GroupManager(datamodel, groupSize);

        File file = new File("data/" + Config.DB_NAME + "/groups-" + groupSize + "-pred-mfgu_bf.csv");

        File parent = file.getAbsoluteFile().getParentFile();
        parent.mkdirs();

        String[] headers = {"mfgu_bf-pred"};
        CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(file), CSVFormat.DEFAULT.withHeader(headers));

        Iterator<Sample> iterator = groupManager.getSamplesIterator();
        while (iterator.hasNext()) {
            Sample sample = iterator.next();

            int itemIndex = sample.getTestItem().getItemIndex();
            Set<TestUser> group = sample.getGroup();

            List<Integer> groupItems = commonRatings(datamodel, group);

            SimpleMatrix S = new SimpleMatrix(groupItems.size(), 1);
            for (int i = 0; i < groupItems.size(); i++) {
                double sum = 0;
                int count = 0;

                for (TestUser testUser : group) {
                   int pos = testUser.findItem(groupItems.get(i));
                   if (pos != -1) {
                       sum += testUser.getRatingAt(pos);
                       count++;
                   }
                }

                double s = sum / count - datamodel.getRatingAverage() - biasedMF.getItemBias(groupItems.get(i));
                S.set(i, 0, s);
            }


            SimpleMatrix A = new SimpleMatrix(groupItems.size(), biasedMF.getNumFactors() + 1);

            A.setColumn(A.numCols() - 1, 0, 1);

            for (int i = 0; i < groupItems.size(); i++) {
                double [] factors = biasedMF.getItemFactors(groupItems.get(i));
                for (int f = 0; f < biasedMF.getNumFactors(); f++) {
                    A.set(i, f, factors[f]);
                }
            }

            SimpleMatrix m = A.transpose()
                                .mult(A)
                                .plus(SimpleMatrix.identity(biasedMF.getNumFactors() + 1).scale(biasedMF.getLambda()))
                                .invert()
                                .mult(A.transpose())
                                .mult(S);

            double groupBias = m.get(biasedMF.getNumFactors(), 0);

            double[] groupFactors = new double[biasedMF.getNumFactors()];
            for (int f = 0; f < biasedMF.getNumFactors(); f++) {
                groupFactors[f] = m.get(f, 0);
            }

            double itemBias = biasedMF.getItemBias(itemIndex);

            double[] itemFactors = biasedMF.getItemFactors(itemIndex);

            double pred = datamodel.getRatingAverage() + + groupBias + itemBias + Maths.dotProduct(groupFactors, itemFactors);
            csvPrinter.print(pred);
            csvPrinter.println();
        }

        csvPrinter.close();
    }

    private static List<Integer> commonRatings(DataModel datamodel, Set<TestUser> group) {
        List<Integer> items = new ArrayList<>();
        for (int itemIndex = 0; itemIndex < datamodel.getNumberOfItems(); itemIndex++) {
            for (TestUser testUser : group) {
                if (testUser.findItem(itemIndex) != -1) {
                    items.add(itemIndex);
                    break;
                }
            }
        }
        return items;
    }

}

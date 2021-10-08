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

import java.io.File;
import java.io.FileWriter;
import java.util.Iterator;

public class MFGU_AF {

    public static void main (String[] args) throws Exception {
        DataModel datamodel = null;

        if (Config.DB_NAME.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        }

        BiasedMF biasedMF = new BiasedMF(datamodel, 6,50, 0.05, 0.01, Config.RANDOM_SEED);
        biasedMF.fit();

        int groupSize = 4;

        GroupManager groupManager = new GroupManager(datamodel, groupSize);

        File file = new File("data/" + Config.DB_NAME + "/groups-" + groupSize + "-pred-mfgu_af.csv");

        File parent = file.getAbsoluteFile().getParentFile();
        parent.mkdirs();

        String[] headers = {"mfgu_af-pred"};
        CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(file), CSVFormat.DEFAULT.withHeader(headers));

        Iterator<Sample> iterator = groupManager.getSamplesIterator();
        while (iterator.hasNext()) {
            Sample sample = iterator.next();

            int itemIndex = sample.getTestItem().getItemIndex();

            double[] itemFactors = biasedMF.getItemFactors(itemIndex);
            double itemBias = biasedMF.getItemBias(itemIndex);

            double[] groupFactors = new double[biasedMF.getNumFactors()];
            double groupBias = 0;

            for (TestUser testUser : sample.getGroup()) {
                int userIndex = testUser.getUserIndex();

                double[] userFactors = biasedMF.getUserFactors(userIndex);
                double userBias = biasedMF.getUserBias(userIndex);

                for (int f = 0; f < biasedMF.getNumFactors(); f++) {
                    groupFactors[f] += userFactors[f] / groupSize;
                    groupBias += userBias / groupSize;
                }
            }

            double pred = datamodel.getRatingAverage() + groupBias + itemBias + Maths.dotProduct(groupFactors, itemFactors);
            csvPrinter.print(pred);
            csvPrinter.println();
        }

        csvPrinter.close();
    }
}

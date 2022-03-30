package com.github.knodis.recommender;

import com.github.knodis.data.GroupManager;
import com.github.knodis.data.Sample;
import com.github.knodis.etc.Config;
import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BiasedMF;
import es.upm.etsisi.cf4j.util.Maths;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.File;
import java.io.FileWriter;
import java.util.Iterator;

public class BiasedMFAverage {

    private final static int[] GROUP_SIZES = {2, 3, 4, 5, 6, 7, 8, 9, 10};

    public static void main (String[] args) throws Exception {
        DataModel datamodel = null;

        if (Config.DB_NAME.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        }

        BiasedMF biasedMF = new BiasedMF(datamodel, 6,50, 0.05, 0.01, Config.RANDOM_SEED);
        biasedMF.fit();

        for (int groupSize : GROUP_SIZES) {
            System.out.println("\nProcessing groups of size " + groupSize);

            GroupManager groupManager = new GroupManager(datamodel, groupSize);

            File file = new File("data/" + Config.DB_NAME + "/groups-" + groupSize + "-biasedmf-avg.csv");

            File parent = file.getAbsoluteFile().getParentFile();
            parent.mkdirs();

            String[] headers = {"biasedmf-avg"};
            CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(file), CSVFormat.DEFAULT.withHeader(headers));

            Iterator<Sample> iterator = groupManager.getSamplesIterator();
            while (iterator.hasNext()) {
                Sample sample = iterator.next();

                int itemIndex = sample.getTestItem().getItemIndex();

                double prediction = 0;

                for (TestUser testUser : sample.getGroup()) {
                    int userIndex = testUser.getUserIndex();
                    prediction += biasedMF.predict(userIndex, itemIndex) / groupSize;
                }

                csvPrinter.print(prediction);
                csvPrinter.println();
            }

            csvPrinter.close();
        }
    }
}

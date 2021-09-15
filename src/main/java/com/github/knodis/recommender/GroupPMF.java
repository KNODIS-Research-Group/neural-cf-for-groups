package com.github.knodis.recommender;

import com.github.knodis.data.GroupManager;
import com.github.knodis.data.Sample;
import com.github.knodis.etc.Config;
import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;
import es.upm.etsisi.cf4j.util.Maths;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.File;
import java.io.FileWriter;
import java.util.Iterator;

public class GroupPMF {

    public static void main (String[] args) throws Exception {
        DataModel datamodel = null;

        if (Config.DB_NAME.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        }

        PMF pmf = new PMF(datamodel, 8,50, 0.045, 0.01, Config.RANDOM_SEED);
        pmf.fit();

        int groupSize = 2;

        GroupManager groupManager = new GroupManager(datamodel, groupSize);

        File file = new File("data/" + Config.DB_NAME + "/groups-" + groupSize + "-pred-pmf.csv");

        File parent = file.getAbsoluteFile().getParentFile();
        parent.mkdirs();

        String[] headers = {"pmf-pred"};
        CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(file), CSVFormat.DEFAULT.withHeader(headers));

        Iterator<Sample> iterator = groupManager.getSamplesIterator();
        while (iterator.hasNext()) {
            Sample sample = iterator.next();

            int itemIndex = sample.getTestItem().getItemIndex();
            double[] itemFactors = pmf.getItemFactors(itemIndex);

            double[] groupFactors = new double[pmf.getNumFactors()];
            for (TestUser testUser : sample.getGroup()) {
                int userIndex = testUser.getUserIndex();
                double[] userFactors = pmf.getUserFactors(userIndex);
                for (int f = 0; f < pmf.getNumFactors(); f++) {
                    groupFactors[f] += userFactors[f] / groupSize;
                }
            }

            double pred = Maths.dotProduct(groupFactors, itemFactors);
            csvPrinter.print(pred);
            csvPrinter.println();
        }

        csvPrinter.close();
    }
}

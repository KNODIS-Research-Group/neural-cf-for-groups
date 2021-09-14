package com.github.knodis.recommender;

import com.github.knodis.etc.Config;
import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Reader;

public class GroupPMF {

    public static void main (String[] args) throws Exception {
        DataModel datamodel = null;

        if (Config.DB_NAME.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        }

        PMF pmf = new PMF(datamodel, 8,10, Config.RANDOM_SEED);
        pmf.fit();

        int groupSize = 2;

        File file = new File("data/" + Config.DB_NAME + "/groups-" + groupSize + "-pred-pmf.csv");

        File parent = file.getAbsoluteFile().getParentFile();
        parent.mkdirs();

        String[] headers = {"pmf-pred"};
        CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(file), CSVFormat.DEFAULT.withHeader(headers));

        Reader in = new FileReader("data/" + Config.DB_NAME + "/groups-" + groupSize + ".csv");
        Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(in);

        for (CSVRecord record : records) {
            String itemId = record.get("item");
            int itemIndex = datamodel.findItemIndex(itemId);

            double sumPred = 0;

            for (int u = 1; u <= groupSize; u++) {
                String userId = record.get("user-" + u);
                int userIndex = datamodel.findUserIndex(userId);

                double pred = pmf.predict(userIndex, itemIndex);
                sumPred += pred;
            }

            double groupPred = sumPred / 2;

            csvPrinter.print(groupPred);
            csvPrinter.println();
        }

        in.close();
        csvPrinter.close();
    }
}

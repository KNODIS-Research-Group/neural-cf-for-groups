package com.github.knodis;

import com.github.knodis.etc.Config;
import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestItem;
import es.upm.etsisi.cf4j.data.TestUser;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.File;
import java.io.FileWriter;
import java.util.*;

public class SampleGroups {

    private final static int NUM_SAMPLES = 1000;
    private final static int GROUP_SIZE = 4;

    public static void main (String[] args) throws Exception {

        Random rand = new Random(Config.RANDOM_SEED);

        DataModel datamodel = null;

        if (Config.DB_NAME.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        }

        File file = new File("data/" + Config.DB_NAME + "/groups-" + GROUP_SIZE + ".csv");

        File parent = file.getAbsoluteFile().getParentFile();
        parent.mkdirs();

        String[] headers = new String[1 + 2 * GROUP_SIZE];
        headers[0] = "item";
        for (int i = 0; i < GROUP_SIZE; i++) {
            headers[1+2*i] = "user-" + (i+1);
            headers[2+2*i] = "rating-" + (i+1);
        }

        CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(file), CSVFormat.DEFAULT.withHeader(headers));

        int samples = 0;
        while (samples < NUM_SAMPLES) {

            if (samples % 100 == 0) System.out.print(".");
            if (samples % 1000 == 0) System.out.println(" " + i + " samples");

            List<String> record = new ArrayList<>();

            int testItemIndex = rand.nextInt(datamodel.getNumberOfTestItems());
            TestItem testItem = datamodel.getTestItem(testItemIndex);

            record.add(testItem.getId());

            Set<Integer> group = new HashSet<>();
            while (group.size() < GROUP_SIZE) {
                int testUserIndex = rand.nextInt(datamodel.getNumberOfTestUsers());
                group.add(testUserIndex);
            }

            boolean isValidGroup = false;

            for (int testUserIndex : group) {
                TestUser testUser = datamodel.getTestUser(testUserIndex);
                record.add(testUser.getId());

                int pos = testUser.findTestItem(testItemIndex);
                if (pos == -1) {
                    record.add("");
                } else {
                    double rating = testUser.getTestRatingAt(pos);
                    record.add(Double.toString(rating));
                    isValidGroup = true;
                }
            }

            if (isValidGroup) {
                csvPrinter.printRecord(record);
                samples++;
            }
        }

        csvPrinter.close();
    }
}

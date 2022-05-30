package com.github.knodis.ncfForGroups;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestItem;
import es.upm.etsisi.cf4j.data.TestUser;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.File;
import java.io.FileWriter;
import java.util.*;

/**
 * This class samples groups for a given dataset and it exports them in a csv file.
 */
public class SampleGroups {

    public final static String DB_NAME = "ml1m";

    public final static Long RANDOM_SEED = 43L;

    private final static int NUM_SAMPLES = 50000;

    private final static int[] GROUP_SIZES = {2, 3, 4, 5, 6, 7, 8, 9, 10};

    public static void main (String[] args) throws Exception {

        Random rand = new Random(RANDOM_SEED);

        DataModel datamodel = null;

        if (DB_NAME.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        } else if (DB_NAME.equals("ft")) {
            datamodel = BenchmarkDataModels.FilmTrust();
        } else if (DB_NAME.equals("anime")) {
            datamodel = BenchmarkDataModels.MyAnimeList();
        }

        for (int groupSize : GROUP_SIZES) {
            System.out.println("\nGenerating groups of size " + groupSize);

            File file = new File("../data/" + DB_NAME + "/groups-" + groupSize + ".csv");

            File parent = file.getAbsoluteFile().getParentFile();
            parent.mkdirs();

            String[] headers = new String[1 + 2 * groupSize];
            headers[0] = "item";
            for (int i = 0; i < groupSize; i++) {
                headers[1+2*i] = "user-" + (i+1);
                headers[2+2*i] = "rating-" + (i+1);
            }

            CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(file), CSVFormat.DEFAULT.withHeader(headers));

            int samples = 0;
            while (samples < NUM_SAMPLES) {

                List<String> record = new ArrayList<>();

                int testItemIndex = rand.nextInt(datamodel.getNumberOfTestItems());
                TestItem testItem = datamodel.getTestItem(testItemIndex);

                if (testItem.getNumberOfTestRatings() >= groupSize) {
                    record.add(Integer.toString(testItem.getItemIndex()));

                    List<Integer> indexes = new ArrayList<>();
                    for (int i = 0; i < testItem.getNumberOfTestRatings(); i++) {
                        indexes.add(i);
                    }
                    Collections.shuffle(indexes);

                    for (int i = 0; i < groupSize; i++) {
                        int pos = indexes.get(i);
                        int testUserIndex = testItem.getTestUserAt(pos);

                        TestUser testUser = datamodel.getTestUser(testUserIndex);
                        record.add(Integer.toString(testUser.getUserIndex()));

                        double rating = testItem.getTestRatingAt(pos);
                        record.add(Double.toString(rating));
                    }

                    csvPrinter.printRecord(record);
                    samples++;

                    if (samples % 100 == 0) System.out.print(".");
                    if (samples % 1000 == 0) System.out.println(" " + samples + " samples");
                }
            }

            csvPrinter.close();
        }
    }
}

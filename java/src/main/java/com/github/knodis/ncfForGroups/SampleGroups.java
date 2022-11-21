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

    public final static String DB_NAME = "anime";

    public final static Long RANDOM_SEED = 43L;

    private final static int NUM_RATINGS = 5; // minimum number of common ratings in a group

    private final static int NUM_GROUPS = 10000; // number of groups

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

        // Build groups of different sizes
        for (int groupSize : GROUP_SIZES) {
            System.out.println("\nGenerating groups of size " + groupSize);

            // Build a list to sample test items that have been rated by more that groupSize users
            List <Integer> testItemIndexes = new ArrayList<>();
            for (int testItemIndex = 0; testItemIndex < datamodel.getNumberOfTestItems(); testItemIndex++) {
                TestItem testItem = datamodel.getTestItem(testItemIndex);
                if (testItem.getNumberOfTestRatings() >= groupSize) {
                    testItemIndexes.add(testItemIndex);
                }
            }

            // Create output file and its headers
            File file = new File("../data/" + DB_NAME + "/groups-" + groupSize + ".csv");

            File parent = file.getAbsoluteFile().getParentFile();
            parent.mkdirs();

            String[] headers = new String[2 + 2 * groupSize];
            headers[0] = "group";
            headers[1] = "item";
            for (int i = 0; i < groupSize; i++) {
                headers[2+2*i] = "user-" + (i+1);
                headers[3+2*i] = "rating-" + (i+1);
            }

            CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(file), CSVFormat.DEFAULT.withHeader(headers));

            // Sample groups
            int groups = 0;
            while (groups < NUM_GROUPS) {

                // get random items
                Collections.shuffle(testItemIndexes, rand);

                // find users that have rated all the items
                Set<Integer> testUserIndexesSet = new HashSet<>();

                for (int i = 0; i < NUM_RATINGS; i++) {
                    int testItemIndex = testItemIndexes.get(i);
                    TestItem testItem = datamodel.getTestItem(testItemIndex);

                    // it is the first user: add all the items
                    if (i == 0) {
                        for (int pos = 0; pos < testItem.getNumberOfTestRatings(); pos++) {
                            int testUserIndex = testItem.getTestUserAt(pos);
                            testUserIndexesSet.add(testUserIndex);
                        }

                    // it is not the first user: retain common items
                    } else {
                        Set<Integer> ratedItemsSet = new HashSet<>();
                        for (int pos = 0; pos < testItem.getNumberOfTestRatings(); pos++) {
                            int testUserIndex = testItem.getTestUserAt(pos);
                            ratedItemsSet.add(testUserIndex);
                        }

                        testUserIndexesSet.retainAll(ratedItemsSet);
                    }
                }

                //System.out.println(testUserIndexesSet.size());

                // sample users
                if (testUserIndexesSet.size() >= groupSize) {

                    List<Integer> testUserIndexesList = new ArrayList<>(testUserIndexesSet); // conversion required to shuffle
                    Collections.shuffle(testUserIndexesList, rand);

                    for (int i = 0; i < NUM_RATINGS; i++) {
                        int testItemIndex = testItemIndexes.get(i);
                        TestItem testItem = datamodel.getTestItem(testItemIndex);

                        List<String> record = new ArrayList<>();

                        record.add(Integer.toString(groups)); // group
                        record.add(Integer.toString(testItem.getItemIndex())); // item

                        for (int g = 0; g < groupSize; g++) {
                            int testUserIndex = testUserIndexesList.get(g);
                            TestUser testUser = datamodel.getTestUser(testUserIndex);

                            record.add(Integer.toString(testUser.getUserIndex())); // user

                            int pos = testUser.findTestItem(testItemIndex);
                            double rating = testUser.getTestRatingAt(pos);
                            record.add(Double.toString(rating)); // rating
                        }

                        csvPrinter.printRecord(record);
                    }

                    groups++;

                    if (groups % 100 == 0) System.out.print(".");
                    if (groups % 1000 == 0) System.out.println(" " + groups + " groups");
                }
            }

            csvPrinter.close();
        }
    }
}

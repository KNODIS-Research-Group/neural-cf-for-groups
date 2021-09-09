package main.java.experiments;

import cf4j.Kernel;
import main.java.data.GroupsManager;

public class BuildFiles {

    // MovieLens 1M

//    final static String DATASET = "movielens1m.txt";
//    final static String SEPARATOR = "::";
//
//    final static double TEST_USERS = 0.2;
//    final static double TEST_ITEMS = 0.3;
//
//    final static int NUM_GROUPS = 1000;
//    final static int NUM_TEST_GROUPS = 200;
//    final static int MIN_GROUP_SIZE = 2;
//    final static int MAX_GROUP_SIZE = 8;
//
//    final static String CF4J_FILE = "ML1M.cf4j";
//    final static String GROUPS_FILE = "GROUPS-ML1M.bin";


    // MovieLens 10M

//    final static String DATASET = "movielens10m.txt";
//    final static String SEPARATOR = "::";
//
//    final static double TEST_USERS = 0.15;
//    final static double TEST_ITEMS = 0.3;
//
//    final static int NUM_GROUPS = 1000;
//    final static int NUM_TEST_GROUPS = 200;
//    final static int MIN_GROUP_SIZE = 2;
//    final static int MAX_GROUP_SIZE = 8;
//
//    final static String CF4J_FILE = "ML10M.cf4j";
//    final static String GROUPS_FILE = "GROUPS-ML10M.bin";


    // FilmTrust

//    final static String DATASET = "FilmTrust.txt";
//    final static String SEPARATOR = " ";
//
//    final static double TEST_USERS = 0.2;
//    final static double TEST_ITEMS = 0.3;
//
//    final static int NUM_GROUPS = 1000;
//    final static int NUM_TEST_GROUPS = 200;
//    final static int MIN_GROUP_SIZE = 2;
//    final static int MAX_GROUP_SIZE = 8;
//
//    final static String CF4J_FILE = "FT.cf4j";
//    final static String GROUPS_FILE = "GROUPS-FT.bin";


    // BookCroosing

//    final static String DATASET = "BookCrossing.csv";
//    final static String SEPARATOR = ";";
//
//    final static double TEST_USERS = 0.2;
//    final static double TEST_ITEMS = 0.3;
//
//    final static int NUM_GROUPS = 1000;
//    final static int NUM_TEST_GROUPS = 200;
//    final static int MIN_GROUP_SIZE = 2;
//    final static int MAX_GROUP_SIZE = 8;
//
//    final static String CF4J_FILE = "BX.cf4j";
//    final static String GROUPS_FILE = "GROUPS-BX.bin";


    // CiaoDVD

//    final static String DATASET = "CiaoDVD.csv";
//    final static String SEPARATOR = ";";
//
//    final static double TEST_USERS = 0.2;
//    final static double TEST_ITEMS = 0.3;
//
//    final static int NUM_GROUPS = 1000;
//    final static int NUM_TEST_GROUPS = 200;
//    final static int MIN_GROUP_SIZE = 2;
//    final static int MAX_GROUP_SIZE = 8;
//
//    final static String CF4J_FILE = "CiaoDVD.cf4j";
//    final static String GROUPS_FILE = "GROUPS-CiaoDVD.bin";


    // Flixter

    final static String DATASET = "Flixter.txt";
    final static String SEPARATOR = ";";

    final static double TEST_USERS = 0.2;
    final static double TEST_ITEMS = 0.3;

    final static int NUM_GROUPS = 1000;
    final static int NUM_TEST_GROUPS = 200;
    final static int MIN_GROUP_SIZE = 2;
    final static int MAX_GROUP_SIZE = 8;

    final static String CF4J_FILE = "Flixter.cf4j";
    final static String GROUPS_FILE = "GROUPS-Flixter.bin";

    public static void main (String [] args) throws Exception {

        Kernel.getInstance().open(DATASET, TEST_USERS, TEST_ITEMS, SEPARATOR);
        Kernel.getInstance().writeKernel(CF4J_FILE);

        GroupsManager.getInstance().generateGroups(NUM_GROUPS, NUM_TEST_GROUPS, MIN_GROUP_SIZE, MAX_GROUP_SIZE);
        GroupsManager.getInstance().write(GROUPS_FILE);
    }
}

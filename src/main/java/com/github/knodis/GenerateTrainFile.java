package com.github.knodis;

import com.github.knodis.etc.Config;
import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.Item;
import es.upm.etsisi.cf4j.data.User;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

public class GenerateTrainFile {

    public static void main (String[] args) throws Exception {

        DataModel datamodel = null;

        if (Config.DB_NAME.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        }

        File file = new File("data/" + Config.DB_NAME + "/training-ratings.csv");

        File parent = file.getAbsoluteFile().getParentFile();
        parent.mkdirs();

        String[] HEADERS = { "user", "item", "rating"};
        CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(file), CSVFormat.DEFAULT.withHeader(HEADERS));

        for (User user : datamodel.getUsers()) {
            for (int pos = 0; pos < user.getNumberOfRatings(); pos++) {
                int itemIndex = user.getItemAt(pos);
                Item item = datamodel.getItem(itemIndex);
                double rating = user.getRatingAt(pos);

                csvPrinter.printRecord(user.getId(), item.getId(), rating);
            }
        }

        csvPrinter.close();

        System.out.println("File " + file.toString() + " generated successfully.");
    }
}

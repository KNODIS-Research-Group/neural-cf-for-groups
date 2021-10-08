package com.github.knodis.data;

import com.github.knodis.etc.Config;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestItem;
import es.upm.etsisi.cf4j.data.TestUser;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class GroupManager {

    private DataModel datamodel;

    private int groupSize;

    public GroupManager(DataModel datamodel, int groupSize) {
        this.datamodel = datamodel;
        this.groupSize = groupSize;
    }

    public Iterator<Sample> getSamplesIterator () throws Exception {

        List<Sample> samples = new ArrayList<>();

        Reader in = new FileReader("data/" + Config.DB_NAME + "/groups-" + this.groupSize + ".csv");
        Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(in);

        for (CSVRecord record : records) {
            int itemIndex = Integer.parseInt(record.get("item"));
            TestItem testItem = (TestItem) this.datamodel.getItem(itemIndex);

            Sample sample = new Sample(testItem);

            for (int u = 1; u <= this.groupSize; u++) {
                int userIndex = Integer.parseInt(record.get("user-" + u));
                TestUser testUser = (TestUser) this.datamodel.getUser(userIndex);
                sample.addTestUser(testUser);
            }

            samples.add(sample);
        }

        return samples.iterator();
    }
}

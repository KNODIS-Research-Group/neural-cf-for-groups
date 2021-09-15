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
import java.util.Set;

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
            String itemId = record.get("item");
            int testItemIndex = this.datamodel.findTestItemIndex(itemId);
            TestItem testItem = this.datamodel.getTestItem(testItemIndex);

            Sample sample = new Sample(testItem);

            for (int u = 1; u <= this.groupSize; u++) {
                String userId = record.get("user-" + u);
                int testUserIndex = this.datamodel.findTestUserIndex(userId);
                TestUser testUser = this.datamodel.getTestUser(testUserIndex);
                sample.addTestUser(testUser);
            }

            samples.add(sample);
        }

        return samples.iterator();
    }
}

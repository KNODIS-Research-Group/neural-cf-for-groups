package com.github.knodis.data;

import es.upm.etsisi.cf4j.data.TestItem;
import es.upm.etsisi.cf4j.data.TestUser;

import java.util.HashSet;
import java.util.Set;

public class Sample {

    private TestItem testItem;

    private Set<TestUser> group;

    public Sample(TestItem testItem) {
        this.testItem = testItem;
        this.group = new HashSet<>();
    }

    public void addTestUser(TestUser testUser) {
        this.group.add(testUser);
    }

    public Set<TestUser> getGroup() {
        return this.group;
    }

    public TestItem getTestItem() {
        return this.testItem;
    }
}

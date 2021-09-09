package main.java.data;

import java.io.*;

public class GroupsManager implements Serializable {

    private static GroupsManager instance;

    private Group [] groups;
    private TestGroup [] testGroups;

    private GroupsManager () { }

    public static GroupsManager getInstance() {
        if (instance == null) {
            instance = new GroupsManager();
        }
        return instance;
    }


    public void generateGroups (int numberOfGroups, int numberOfTestGroups, int minSize, int maxSize) {

        this.groups = new Group [numberOfGroups];
        this.testGroups = new TestGroup [numberOfTestGroups];

        for (int g = 0; g < numberOfTestGroups; g++) {
            int size = this.getRandomSize(minSize, maxSize);
            this.testGroups[g] = new TestGroup(size, g);
            this.groups[g] = this.testGroups[g];
        }

        for (int g = numberOfTestGroups; g < numberOfGroups; g++) {
            int size = this.getRandomSize(minSize, maxSize);
            this.groups[g] = new Group(size, g);
        }
    }

    public int getNumberOfGroups () {
        return this.groups.length;
    }

    public int getNumberOfTestGroups () {
        return this.testGroups.length;
    }

    public Group [] getGroups () {
        return this.groups;
    }

    public TestGroup [] getTestGroups () {
        return this.testGroups;
    }

    public Group getGroupByIndex (int groupIndex) {
        return this.groups[groupIndex];
    }

    public TestGroup getTestGroupByIndex (int testGroupIndex) {
        return this.testGroups[testGroupIndex];
    }

    private int getRandomSize (int minSize, int maxSize) {
        return (int) Math.floor(Math.random() * (maxSize + 1 - minSize) + minSize);
    }

    public void write (String filename) throws Exception {
        ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(filename)));
        oos.writeObject(this);
        oos.flush();
        oos.close();
    }

    public static void read (String filename) throws Exception {
        ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filename)));
        GroupsManager.instance = (GroupsManager) ois.readObject();
        ois.close();
    }

}

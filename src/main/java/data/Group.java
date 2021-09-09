package main.java.data;

import java.io.Serializable;
import java.util.*;

import cf4j.Item;
import cf4j.Kernel;
import cf4j.TestUser;
import cf4j.User;

public class Group implements Serializable {

    protected Set <Integer> usersIndexes;

    protected int groupIndex;

    // Required by heritage relationship
    // Group.setUsersIndexes() method should be called after this constructor
    protected Group (int groupIndex) {
        this.groupIndex = groupIndex;
    }

    public Group (int size, int groupIndex) {
        this.groupIndex = groupIndex;
        this.usersIndexes = new HashSet<>();

        while (this.usersIndexes.size() < size) {
            int randomIndex = (int) (Math.random() * Kernel.getInstance().getNumberOfUsers());
            this.usersIndexes.add(randomIndex);
        }
    }

    protected void setUsersIndexes (Set <Integer> usersIndexes) {
        this.usersIndexes = usersIndexes;
    }

    public Collection<Integer> getUsersIndexes () {
        return this.usersIndexes;
    }

    public boolean hasRated(int itemCode){
        Iterator <Double> iter = this.getRatings(itemCode);
        return iter.hasNext();
    }

    public Iterator <Double> getRatings (int itemCode) {
        List<Double> ratings = new ArrayList<>();

        for (int userIndex : this.usersIndexes) {
            User user = Kernel.getInstance().getUserByIndex(userIndex);
            int i = user.getItemIndex(itemCode);
            if (i != -1) {
                ratings.add(user.getRatingAt(i));
            }
        }

        return ratings.iterator();
    }

    public int getGroupIndex () {
        return this.groupIndex;
    }
}

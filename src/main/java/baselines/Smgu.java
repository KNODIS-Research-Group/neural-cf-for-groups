package main.java.baselines;

import cf4j.Item;
import cf4j.Kernel;
import cf4j.User;
import cf4j.utils.Methods;
import main.java.data.Group;
import main.java.data.GroupsManager;
import main.java.gumf.GroupRecommender;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Ortega, F., Hurtado, R., Bobadilla, J., & Bojorque, R. (2018). Recommendation to groups of users using the singularities concept. IEEE Access, 6, 39745-39761.
 */
public class Smgu implements GroupRecommender {

    // hyper-parameters
    private double alpha;
    private int numNeighbors;

    // parameters
    private int [][] neighbors;
    private double [][] neighborsSimilarities;


    public Smgu (double alpha, int numNeighbors) {

        // set hyper-parameters
        this.alpha = alpha;
        this.numNeighbors = numNeighbors;

        // initialize neighbors
        int numGroups = GroupsManager.getInstance().getNumberOfGroups();

        this.neighbors = new int[numGroups][numNeighbors];
        this.neighborsSimilarities = new double[numGroups][numNeighbors];

        for (int i = 0; i < this.neighbors.length; i++) {
            for (int j = 0; j < this.neighbors[i].length; j++) {
                this.neighbors[i][j] = -1;
                this.neighborsSimilarities[i][j] = Double.NEGATIVE_INFINITY;
            }
        }
    }

    public void fit () {

        System.out.println("\nComputing SMGU...");

        int numUsers = Kernel.getInstance().getNumberOfUsers();
        int numItems = Kernel.getInstance().getNumberOfItems();
        int numGroups = GroupsManager.getInstance().getNumberOfGroups();


        // compute p_i
        double [] pi = new double[numItems];
        for (int i = 0; i < pi.length; i++) {
            Item item = Kernel.getInstance().getItemByIndex(i);
            pi[i] = 1 - (double) item.getNumberOfRatings() / Kernel.getInstance().getNumberOfUsers();
        }


        // compute singularities
        Map<Double, Double>[] singularities = new Map[numItems];
        for (int i = 0; i < singularities.length; i++) {
            singularities[i] = new HashMap<>();

            Item item = Kernel.getInstance().getItemByIndex(i);
            for (double rui : item.getRatings()) {
                if (!singularities[i].containsKey(rui)) {
                    int count = 0;
                    for (double rvi : item.getRatings()) {
                        if (rui == rvi) count++;
                    }

                    double sui = (double) count / item.getNumberOfRatings();
                    singularities[i].put(rui, sui);
                }
            }
        }


        // compute neighbors
        for (int groupIndex = 0; groupIndex < numGroups; groupIndex++) {
            Group group = GroupsManager.getInstance().getGroupByIndex(groupIndex);

            double [] similarities = new double [numUsers];

            for (int userIndex = 0; userIndex < numUsers; userIndex++) {
                User user = Kernel.getInstance().getUserByIndex(userIndex);

                // user belongs to the group
                if (group.getUsersIndexes().contains(userIndex)) {
                    similarities[userIndex] = Double.NEGATIVE_INFINITY;

                // user does not belong to the group
                } else {
                    double xgu = this.computeXgu(pi, group, user);
                    double ygu = this.computeYgu(singularities, group, user);

                    similarities[userIndex] = Math.pow(xgu, this.alpha) * Math.pow(ygu, 1 - this.alpha);

                    if (Double.isNaN(similarities[userIndex])) {
                        similarities[userIndex] = Double.NEGATIVE_INFINITY;
                    }
                }
            }

            this.neighbors[groupIndex] = Methods.findTopN(similarities, this.numNeighbors);

            for (int n = 0; n < this.numNeighbors; n++) {
                int userIndex = this.neighbors[groupIndex][n];
                if (userIndex == -1) break;
                this.neighborsSimilarities[groupIndex][n] = similarities[userIndex];
            }

            if ((groupIndex + 1) % 10 == 0) System.out.print(".");
            if ((groupIndex + 1) % 100 == 0) System.out.println((groupIndex + 1) + " groups");
        }
    }

    private double computeXgu (double [] pi, Group group, User user) {
        double num = 0;
        double den = 0;

        for (int i = 0; i < Kernel.getInstance().getNumberOfItems(); i++) {
            Item item = Kernel.getInstance().getItemByIndex(i);
            int itemCode = item.getItemCode();

            boolean ratedByGroup = group.hasRated(itemCode);
            boolean ratedByUser = user.getItemIndex(itemCode) != -1;

            if (ratedByGroup && ratedByUser) num += pi[i];
            if (ratedByGroup || ratedByUser) den += pi[i];
        }

        return num / den;
    }

    private double computeYgu (Map<Double, Double>[] singularities, Group group, User user) {
        double maxDiff = Kernel.getInstance().getMaxRating() - Kernel.getInstance().getMinRating();

        double num = 0;
        double den = 0;

        for (int i = 0; i < user.getNumberOfRatings(); i++) {
            int itemCode = user.getItemAt(i);
            double userRating = user.getRatingAt(i);

            if (group.hasRated(itemCode)) {

                double sui = this.getSingularity(singularities, itemCode, userRating);
                double sGi = 1;

                double diff = 0; // A_{G,u,i}
                int count = 0; // #G_i

                Iterator<Double> iter = group.getRatings(itemCode);
                while (iter.hasNext()) {
                    double groupRating = iter.next();

                    diff += Math.pow(userRating - groupRating, 2);
                    sGi *= this.getSingularity(singularities, itemCode, groupRating);
                    count++;
                }

                sGi = Math.pow(sGi, 1.0 / count);
                diff = diff / (Math.pow(maxDiff, 2) * count);

                num += sui * sGi * count * diff;
                den += sui * sGi * count;
            }
        }

        return 1.0 - (num / den);
    }

    private double getSingularity (Map<Double, Double>[] singularities, int itemCode, double rating) {
        int itemIndex = Kernel.getInstance().getItemIndex(itemCode);
        double singularity = singularities[itemIndex].get(rating);
        return singularity;
    }

    @Override
    public double getGroupPrediction(int groupIndex, int itemIndex) {
        Item item = Kernel.getInstance().getItemByIndex(itemIndex);
        int itemCode = item.getItemCode();

        double num = 0;
        double den = 0;

        for (int n = 0; n < this.numNeighbors; n++) {
            int userIndex = this.neighbors[groupIndex][n];

            if (userIndex != -1) {
                User user = Kernel.getInstance().getUserByIndex(userIndex);
                int i = user.getItemIndex(itemCode);
                if (i != -1) {
                    double sim = this.neighborsSimilarities[groupIndex][n];
                    num += sim * user.getRatingAt(i);
                    den += sim;
                }
            }
        }

        return (den > 0) ? num / den : Double.NEGATIVE_INFINITY;
    }
}

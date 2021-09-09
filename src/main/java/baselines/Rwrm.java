package main.java.baselines;

import cf4j.Item;
import cf4j.Kernel;
import main.java.data.Group;
import main.java.data.GroupsManager;
import main.java.gumf.GroupRecommender;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

/**
 * Feng, S., & Cao, J. (2017). Improving group recommendations via detecting comprehensive correlative information. Multimedia Tools and Applications, 76(1), 1355-1377.
 */
public class Rwrm implements GroupRecommender {

    private double rho;

    private int numIterations;

    private SimpleMatrix [] c_gim;

    public Rwrm (double rho, int numIterations) {
        this.rho = rho;
        this.numIterations = numIterations;
        this.c_gim = new SimpleMatrix [GroupsManager.getInstance().getNumberOfTestGroups()];
    }

    private SimpleMatrix columNormalize (SimpleMatrix m) {
        SimpleMatrix l = new SimpleMatrix(m);

        for (int c = 0; c < m.numCols(); c++) {
            double sum = 0;

            for (int r = 0; r < m.numRows(); r++) {
                sum += l.get(r, c);
            }

            if (sum != 0) {
                for (int r = 0; r < m.numRows(); r++) {
                    double value = l.get(r, c) / sum;
                    l.set(r, c, value);
                }
            }
        }

        return l;
    }

    public void fit () {

        int numUsers = Kernel.getInstance().getNumberOfUsers();
        int numItems = Kernel.getInstance().getNumberOfItems();
        int numGroups = GroupsManager.getInstance().getNumberOfGroups();

        // compute matrix that connects groups with items
        SimpleMatrix x_gm = new SimpleMatrix(numGroups, numItems);
        x_gm.set(0);

        for (int groupIndex = 0; groupIndex < numGroups; groupIndex++) {
            Group group = GroupsManager.getInstance().getGroupByIndex(groupIndex);

            for (int itemIndex = 0; itemIndex < numItems; itemIndex++) {
                Item item = Kernel.getInstance().getItemByIndex(itemIndex);
                int itemCode = item.getItemCode();

                if (group.hasRated(itemCode)) {
                    x_gm.set(groupIndex, itemIndex, 1);
                }
            }
        }

        // compute matrix that connects users with groups
        SimpleMatrix x_ug = new SimpleMatrix(numUsers, numGroups);
        x_ug.set(0);

        for (int groupIndex = 0; groupIndex < numGroups; groupIndex++) {
            Group group = GroupsManager.getInstance().getGroupByIndex(groupIndex);

            for (int userIndex : group.getUsersIndexes()) {
                x_ug.set(userIndex, groupIndex, 1);
            }
        }

        // compute matrix that connects items with users
        SimpleMatrix x_mu = new SimpleMatrix(numItems, numUsers);
        x_mu.set(0);

        for (int itemIndex = 0; itemIndex < numItems; itemIndex++) {
            Item item = Kernel.getInstance().getItemByIndex(itemIndex);

            for (int userCode : item.getUsers()) {
                int userIndex = Kernel.getInstance().getUserIndex(userCode);
                x_mu.set(itemIndex, userIndex, 1);

            }
        }

        // normalize cols
        SimpleMatrix l_gm = this.columNormalize(x_gm);
        SimpleMatrix l_mug = this.columNormalize(x_mu.mult(x_ug));

        // fit for each group
        for (Group group : GroupsManager.getInstance().getTestGroups()) {
            int groupIndex = group.getGroupIndex();

            SimpleMatrix pi_i = new SimpleMatrix(numGroups + numItems, 1);
            pi_i.set(0);
            pi_i.set(groupIndex, 0, 1 - this.rho);

            this.c_gim[groupIndex] = SimpleMatrix.random64(numGroups + numItems, 1, 0, 1, new Random());

            for (int epoch = 0; epoch < this.numIterations; epoch++) {

                SimpleMatrix c_gim_m = new SimpleMatrix(numItems, 1);
                for (int i = 0; i < numItems; i++) {
                    double value = this.c_gim[groupIndex].get(numGroups + i, 0);
                    c_gim_m.set(i, 0, value);
                }

                SimpleMatrix aux_g = l_gm.mult(c_gim_m);

                SimpleMatrix c_gim_g = new SimpleMatrix(numGroups, 1);
                for (int g = 0; g < numGroups; g++) {
                    double value = this.c_gim[groupIndex].get(g, 0);
                    c_gim_g.set(g, 0, value);
                }

                SimpleMatrix aux_m = l_mug.mult(c_gim_g);

                SimpleMatrix aux = new SimpleMatrix(numGroups + numItems, 1);

                for (int g = 0; g < numGroups; g++) {
                    double value = aux_g.get(g, 0);
                    aux.set(g, 0, value);
                }

                for (int i = 0; i < numItems; i++) {
                    double value = aux_m.get(i, 0);
                    aux.set(numGroups + i, 0, value);
                }

                this.c_gim[groupIndex] = aux.scale(this.rho).plus(pi_i);
            }
        }
    }


    @Override
    public double getGroupPrediction(int groupIndex, int itemIndex) {
        int numGroups = GroupsManager.getInstance().getNumberOfGroups();
        double prediction = this.c_gim[groupIndex].get(numGroups + itemIndex, 0);
        return prediction;
    }
}

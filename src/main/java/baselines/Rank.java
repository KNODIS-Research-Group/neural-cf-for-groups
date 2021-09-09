package main.java.baselines;

import cf4j.*;
import cf4j.knn.userToUser.neighbors.Neighbors;
import cf4j.knn.userToUser.similarities.MetricCorrelation;
import cf4j.utils.Methods;
import main.java.data.GroupsManager;
import main.java.data.TestGroup;
import main.java.gumf.GroupRecommender;

import java.util.HashMap;
import java.util.Map;

/**
 * Baltrunas, L., Makcinskas, T., & Ricci, F. (2010, September). Group recommendations with rank aggregation and collaborative filtering. In Proceedings of the fourth ACM conference on Recommender systems (pp. 119-126). ACM.
 */
public class Rank implements GroupRecommender {

    private int numNeighbors;

    private double [][] scores;

    private Map<Integer, Integer> indexToTestIndex;

    public Rank(int numNeighbors) {
        this.numNeighbors = numNeighbors;
    }

    public void fit() {

        // Get neighbors
        Processor.getInstance().testUsersProcess(new MetricCorrelation(), false);
        Processor.getInstance().testUsersProcess(new Neighbors(this.numNeighbors), false);
        Processor.getInstance().testUsersProcess(new DeviationFromMean(), false);

        // compute borda scores
        int numTestGroups = GroupsManager.getInstance().getNumberOfTestGroups();
        int numTestItems = Kernel.getInstance().getNumberOfTestItems();

        this.scores = new double [numTestGroups][numTestItems];
        this.indexToTestIndex = new HashMap<Integer, Integer>();

        for (TestGroup testGroup : GroupsManager.getInstance().getTestGroups()) {
            int testGroupIndex = testGroup.getTestGroupIndex();
            int groupIndex = testGroup.getGroupIndex();

            this.indexToTestIndex.put(groupIndex, testGroupIndex);

            for (int testUserIndex : testGroup.getTestUsersIndexes()) {
                TestUser groupUser = Kernel.getInstance().getTestUserByIndex(testUserIndex);

                double [] predictions = groupUser.getPredictions();
                int [] recomendations = Methods.findTopN(predictions, numTestItems);

                for (int rank = 0; rank < numTestItems; rank++) {
                    int testItemIndex = recomendations[rank];
                    if (testItemIndex == -1) break;
                    this.scores[testGroupIndex][testItemIndex] += numTestItems - rank + 1;
                }
            }
        }
    }

    @Override
    public double getGroupPrediction(int groupIndex, int itemIndex) {
        int testGroupIndex = this.indexToTestIndex.get(groupIndex);

        Item item = Kernel.getInstance().getItemByIndex(itemIndex);
        int itemCode = item.getItemCode();
        int testItemIndex = Kernel.getInstance().getTestItemIndex(itemCode);

        return this.scores[testGroupIndex][testItemIndex];
    }

    private class DeviationFromMean implements TestUsersPartible {

        private double minSim;

        private double maxSim;

        @Override
        public void beforeRun() {
            this.maxSim = Double.MIN_VALUE;
            this.minSim = Double.MAX_VALUE;

            for (TestUser testUser : Kernel.gi().getTestUsers()) {
                for (double m : testUser.getSimilarities()) {
                    if (!Double.isInfinite(m)) {
                        if (m < this.minSim) this.minSim = m;
                        if (m > this.maxSim) this.maxSim = m;
                    }
                }
            }
        }

        @Override
        public void run (int testUserIndex) {

            TestUser testUser = Kernel.getInstance().getTestUserByIndex(testUserIndex);

            int [] neighbors = testUser.getNeighbors();
            double [] similarities = testUser.getSimilarities();

            int numRatings = Kernel.getInstance().getNumberOfTestItems();
            double [] predictions = new double [numRatings];

            for (int testItemIndex = 0; testItemIndex < numRatings; testItemIndex++) {

                TestItem testItem = Kernel.getInstance().getTestItemByIndex(testItemIndex);
                int itemCode = testItem.getItemCode();

                double sumSimilarities = 0;

                for (int n = 0; n < neighbors.length; n++) {
                    if (neighbors[n] == -1) break; // Neighbors array is filled with -1 when no more neighbors exists

                    int userIndex = neighbors[n];
                    User neighbor = Kernel.getInstance().getUserByIndex(userIndex);

                    int i = neighbor.getItemIndex(itemCode);
                    if (i != -1) {
                        double similarity = similarities[userIndex];
                        double sim = (similarity - this.minSim) / (this.maxSim - this.minSim);

                        predictions[testItemIndex] += sim * (neighbor.getRatings()[i] - neighbor.getRatingAverage());
                        sumSimilarities += sim;
                    }
                }

                if (sumSimilarities == 0) {
                    predictions[testItemIndex] = Double.NaN;
                }
                else {
                    double deviation = predictions[testItemIndex] / sumSimilarities;
                    double prediction = testUser.getRatingAverage() + deviation;
                    prediction = Math.min(prediction, Kernel.gi().getMaxRating());
                    prediction = Math.max(prediction, Kernel.gi().getMinRating());

                    predictions[testItemIndex] = prediction;
                }
            }

            testUser.setPredictions(predictions);
        }

        @Override
        public void afterRun() { }
    }
}

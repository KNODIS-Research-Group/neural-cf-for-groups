package main.java.qualityMeasures;

import cf4j.Item;
import cf4j.Kernel;
import cf4j.TestItem;
import cf4j.User;
import cf4j.utils.Methods;
import main.java.data.Group;
import main.java.data.GroupsManager;
import main.java.data.TestGroup;
import main.java.gumf.GroupRecommender;

import java.util.Iterator;
import java.util.function.Function;

public class GroupQualityMeasures {

    public static double getPrecision (GroupRecommender recommender, int numberOfRecommendations, double relevantThreshold, double phi) {

        double precision = 0;
        int count = 0;

        for (int index = 0; index < GroupsManager.getInstance().getNumberOfTestGroups(); index++) {
            TestGroup testGroup = GroupsManager.getInstance().getTestGroupByIndex(index);
            int [] recommendations = testGroup.getRecommendations(recommender, numberOfRecommendations);

            int success = 0;
            int total = 0;

            for (int testItemIndex : recommendations) {
                if (testItemIndex == -1) break;

                TestItem testItem = Kernel.getInstance().getTestItemByIndex(testItemIndex);
                int itemCode = testItem.getItemCode();

                int numLikes = 0;
                int numDislikes = 0;

                Iterator<Double> iter = testGroup.getTestRatings(itemCode);
                while (iter.hasNext()) {
                    double rating = iter.next();
                    if (rating >= relevantThreshold) {
                        numLikes++;
                    } else {
                        numDislikes++;
                    }
                }

                if (numLikes + numDislikes > 0) {
                    double ratio = (double) numLikes / (numLikes + numDislikes);
                    if (ratio >= phi) success++;
                    total++;
                }
            }

            if (total > 0) {
                precision += (double) success / total;
                count++;
            }
        }

        return precision / count;
    }

    public static double getRecall (GroupRecommender recommender, int numberOfRecommendations, double relevantThreshold, double phi) {

        double recall = 0;
        int count = 0;

        for (int index = 0; index < GroupsManager.getInstance().getNumberOfTestGroups(); index++) {
            TestGroup testGroup = GroupsManager.getInstance().getTestGroupByIndex(index);
            int [] recommendations = testGroup.getRecommendations(recommender, numberOfRecommendations);

            int success = 0;

            for (int testItemIndex : recommendations) {
                if (testItemIndex == -1) break;

                TestItem testItem = Kernel.getInstance().getTestItemByIndex(testItemIndex);
                int itemCode = testItem.getItemCode();

                int numLikes = 0;
                int numDislikes = 0;

                Iterator<Double> iter = testGroup.getTestRatings(itemCode);
                while (iter.hasNext()) {
                    double rating = iter.next();
                    if (rating >= relevantThreshold) {
                        numLikes++;
                    } else {
                        numDislikes++;
                    }
                }

                if (numLikes + numDislikes > 0) {
                    double ratio = (double) numLikes / (numLikes + numDislikes);
                    if (ratio >= phi) success++;
                }
            }

            int total = 0;

            for (int testItemIndex = 0; testItemIndex < Kernel.getInstance().getNumberOfTestItems(); testItemIndex++) {

                TestItem testItem = Kernel.getInstance().getTestItemByIndex(testItemIndex);
                int itemCode = testItem.getItemCode();

                int numLikes = 0;
                int numDislikes = 0;

                Iterator<Double> iter = testGroup.getTestRatings(itemCode);
                while (iter.hasNext()) {
                    double rating = iter.next();
                    if (rating >= relevantThreshold) {
                        numLikes++;
                    } else {
                        numDislikes++;
                    }
                }

                if (numLikes + numDislikes > 0) {
                    double ratio = (double) numLikes / (numLikes + numDislikes);
                    if (ratio >= phi) total++;
                }
            }

            if (total > 0) {
                recall += (double) success / total;
                count++;
            }
        }

        return recall / count;
    }

    public static double getNdcg (GroupRecommender recommender, int numberOfRecommendations, Function <Iterator <Double>, Double> merger) {

        double ndcg = 0;
        int count = 0;

        for (int index = 0; index < GroupsManager.getInstance().getNumberOfTestGroups(); index++) {
            TestGroup testGroup = GroupsManager.getInstance().getTestGroupByIndex(index);

            double [] testRatings = testGroup.mergeTestRatings(merger);
            double [] predictions = testGroup.getPredictions(recommender);

            for (int i = 0; i < Kernel.getInstance().getNumberOfTestItems(); i++) {
                TestItem item = Kernel.getInstance().getTestItemByIndex(i);
                int itemCode = item.getItemCode();

                if (!testGroup.hasTestRated(itemCode)) {
                    testRatings[i] = Double.NEGATIVE_INFINITY;
                    predictions[i] = Double.NEGATIVE_INFINITY;
                }
            }


            // Compute dcg

            int [] recommendations = Methods.findTopN(predictions, numberOfRecommendations);

            double dcg = 0d;

            for (int i = 0; i < recommendations.length; i++) {
                int testItemIndex = recommendations[i];
                if (testItemIndex == -1) break;

                int itemCode = Kernel.getInstance().getTestItemByIndex(testItemIndex).getItemCode();
                Iterator <Double> ratings = testGroup.getTestRatings(itemCode);
                if (!ratings.hasNext()) continue; // item has not been rated

                double rating = merger.apply(ratings);

                dcg += (Math.pow(2, rating) - 1) / (Math.log(i + 2) / Math.log(2));
            }


            // Compute idcg

            int [] idealRecommendations = Methods.findTopN(testRatings, numberOfRecommendations);

            double idcg = 0d;

            for (int i = 0; i < idealRecommendations.length; i++) {
                int testItemIndex = idealRecommendations[i];
                if (testItemIndex == -1) break;

                int itemCode = Kernel.getInstance().getTestItemByIndex(testItemIndex).getItemCode();
                Iterator <Double> ratings = testGroup.getTestRatings(itemCode);
                if (!ratings.hasNext()) continue; // item has not been rated

                double rating = merger.apply(ratings);

                idcg += (Math.pow(2, rating) - 1) / (Math.log(i + 2) / Math.log(2));
            }


            // average results

            if (idcg > 0) {
                ndcg = (count * ndcg + dcg / idcg) / (count + 1);
                count++;
            }
        }

        return ndcg;
    }

    public static double getNormalizedRmse (GroupRecommender recommender, Function<Iterator<Double>, Double> merger) {

        double userRatingAvg = getUserRatingAverage();
        double userRatingStd = getUserRatingStrandardDeviation(userRatingAvg);

        double groupRatingAvg = getGroupRatingAverage(merger);
        double groupRatingStd = getGroupRatingStandardDeviation(merger, groupRatingAvg);

        double sum = 0;
        int count = 0;

        for (int groupIndex = 0; groupIndex < GroupsManager.getInstance().getNumberOfTestGroups(); groupIndex++) {
            TestGroup testGroup = GroupsManager.getInstance().getTestGroupByIndex(groupIndex);

            for (Item item : Kernel.getInstance().getTestItems()) {
                int itemIndex = item.getItemIndex();
                int itemCode = item.getItemCode();

                double prediction = recommender.getGroupPrediction(groupIndex, itemIndex);
                if (Double.isNaN(prediction) || Double.isInfinite(prediction)) continue;

                double normPrediction = (prediction - groupRatingAvg) / groupRatingStd;

                Iterator<Double> iter = testGroup.getTestRatings(itemCode);
                while (iter.hasNext()) {
                    double normUserRating = (iter.next() - userRatingAvg) / userRatingStd;
                    double diff = normPrediction - normUserRating;
                    sum += Math.pow(diff, 2);
                    count++;
                }
            }
        }

        return Math.sqrt(sum / count);
    }

    private static double getUserRatingAverage () {
        double sum = 0;
        int count = 0;

        for (User user : Kernel.getInstance().getUsers()) {
            for (double rating : user.getRatings()) {
                sum += rating;
                count++;
            }
        }

        return sum / count;
    }

    private static double getUserRatingStrandardDeviation (double avg) {
        double sum = 0;
        int count = 0;

        for (User user : Kernel.getInstance().getUsers()) {
            for (double rating : user.getRatings()) {
                sum += Math.pow(rating - avg, 2);
                count++;
            }
        }

        return Math.sqrt(sum / count);
    }

    private static double getGroupRatingAverage (Function<Iterator<Double>, Double> merger) {
        double sum = 0;
        int count = 0;

        for (Group group : GroupsManager.getInstance().getGroups()) {
            for (Item item : Kernel.getInstance().getItems()) {
                int itemCode = item.getItemCode();

                Iterator<Double> ratings = group.getRatings(itemCode);
                if (ratings.hasNext()) {
                    double rating = merger.apply(ratings);
                    sum += rating;
                    count++;
                }
            }
        }

        return sum / count;
    }

    private static double getGroupRatingStandardDeviation (Function<Iterator<Double>, Double> merger, double avg) {
        double sum = 0;
        int count = 0;

        for (Group group : GroupsManager.getInstance().getGroups()) {
            for (Item item : Kernel.getInstance().getItems()) {
                int itemCode = item.getItemCode();

                Iterator<Double> ratings = group.getRatings(itemCode);
                if (ratings.hasNext()) {
                    double rating = merger.apply(ratings);
                    sum += Math.pow(rating - avg, 2);
                    count++;
                }
            }
        }

        return Math.sqrt(sum / count);
    }
}

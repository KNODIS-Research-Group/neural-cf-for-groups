package main.java.data;

import java.util.*;
import java.util.function.Function;

import cf4j.*;
import cf4j.utils.Methods;
import main.java.gumf.GroupRecommender;

public class TestGroup extends Group {

	protected Set <Integer> testUsersIndexes;

	protected int testGroupIndex;

	public TestGroup (int size, int index) {
		this(size, index, index);
	}

	public TestGroup (int size, int groupIndex, int testGroupIndex) {
		super(groupIndex);

		this.testGroupIndex = testGroupIndex;

		this.testUsersIndexes = new HashSet<>();
		Set <Integer> usersIndexes = new HashSet<>();

		while (this.testUsersIndexes.size() < size) {
			int randomIndex = (int) (Math.random() * Kernel.getInstance().getNumberOfTestUsers());
			this.testUsersIndexes.add(randomIndex);

			TestUser testUser = Kernel.getInstance().getTestUserByIndex(randomIndex);
			int userIndex = testUser.getUserIndex();
			usersIndexes.add(userIndex);
		}

		super.setUsersIndexes(usersIndexes);
	}

	public int getTestGroupIndex () {
		return this.testGroupIndex;
	}

	public Collection<Integer> getTestUsersIndexes () {
		return this.testUsersIndexes;
	}

	public boolean hasTestRated(int itemCode){
		Iterator <Double> iter = this.getTestRatings(itemCode);
		return iter.hasNext();
	}

	public Iterator <Double> getTestRatings (int itemCode) {
		List<Double> ratings = new ArrayList<>();

		for (int testUserIndex : this.testUsersIndexes) {
			TestUser user = Kernel.getInstance().getTestUserByIndex(testUserIndex);
			int i = user.getTestItemIndex(itemCode);
			if (i != -1) {
				ratings.add(user.getTestRatingAt(i));
			}
		}

		return ratings.iterator();
	}

	public double [] mergeTestRatings (Function<Iterator<Double>, Double> merger) {
		double [] ratings = new double [Kernel.getInstance().getNumberOfTestItems()];

		for (int i = 0; i < Kernel.getInstance().getNumberOfTestItems(); i++) {
			TestItem item = Kernel.getInstance().getTestItemByIndex(i);
			int itemCode = item.getItemCode();

			Iterator <Double> iter = this.getTestRatings(itemCode);
			ratings[i] = merger.apply(iter);
		}

		return ratings;
	}

	public double [] getPredictions (GroupRecommender recommender) {
		double [] predictions = new double [Kernel.getInstance().getNumberOfTestItems()];

		for (int i = 0; i < Kernel.getInstance().getNumberOfTestItems(); i++) {
			TestItem testItem = Kernel.getInstance().getTestItemByIndex(i);
			int itemIndex = testItem.getItemIndex();
			predictions[i] = recommender.getGroupPrediction(super.getGroupIndex(), itemIndex);
		}

		return predictions;
	}

	public int [] getRecommendations (GroupRecommender recommender, int numberOfRecommendations) {
		double [] predictions = this.getPredictions(recommender);
		int [] recommendations = Methods.findTopN(predictions, numberOfRecommendations);
		return recommendations;
	}
}

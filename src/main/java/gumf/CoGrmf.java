package main.java.gumf;

import cf4j.Item;
import cf4j.Kernel;
import cf4j.model.matrixFactorization.FactorizationModel;
import cf4j.utils.Methods;
import main.java.data.Group;
import main.java.data.GroupsManager;

import java.util.Iterator;
import java.util.function.Function;

public class CoGrmf implements FactorizationModel, GroupRecommender {

	private Function<Iterator<Double>, Double> merger;

	// model parameters

	private double [][] usersFactors;
	private double [][] itemsFactors;
	private double [][] groupsFactors;


	// model hiper-paramenters

	private int numFactors;
	private int numIters;
	private double learningRate;
	private double regularization;


	public CoGrmf(int numFactors, int numIters, double regularization, double learningRate, Function<Iterator<Double>, Double> merger) {
		this.merger = merger;

		this.numFactors = numFactors;
		this.numIters = numIters;
		this.regularization = regularization;
		this.learningRate = learningRate;

		this.usersFactors = new double [Kernel.getInstance().getNumberOfUsers()][numFactors];
		for (int userIndex = 0; userIndex < Kernel.getInstance().getNumberOfUsers(); userIndex++) {
			this.setUserFactors(userIndex, this.random(this.numFactors, -1, 1));
		}

		this.itemsFactors = new double [Kernel.getInstance().getNumberOfItems()][numFactors];
		for (int itemIndex = 0; itemIndex < Kernel.getInstance().getNumberOfItems(); itemIndex++) {
			this.setItemFactors(itemIndex, this.random(this.numFactors, -1, 1));
		}

		this.groupsFactors = new double [GroupsManager.getInstance().getNumberOfGroups()][numFactors];
		for (int groupIndex = 0; groupIndex < GroupsManager.getInstance().getNumberOfGroups(); groupIndex++) {
			this.setGroupFactors(groupIndex, this.random(this.numFactors, -1, 1));
		}
	}



	public void train () {

		System.out.println("\nProcessing CoGrmf...");

		for (int iter = 1; iter <= this.numIters; iter++) {

			System.out.print("iter " + iter + " of " + this.numIters + "...");

			for (int itemIndex = 0; itemIndex < Kernel.getInstance().getNumberOfItems(); itemIndex++) {

				Item item = Kernel.getInstance().getItemByIndex(itemIndex);
				int itemCode = item.getItemCode();

				for (int u = 0; u < item.getNumberOfRatings(); u++) {

					int userCode = item.getUserAt(u);
					int userIndex = Kernel.getInstance().getUserIndex(userCode);

					double rating = item.getRatingAt(u);
					double prediction = this.getPrediction(userIndex, itemIndex);

					double error = rating - prediction;

					// update factors

					double [] pu = this.getUserFactors(userIndex);
					double [] qi = this.getItemFactors(itemIndex);

					for (int k = 0; k < this.numFactors; k++) {
						double dpuk = error * qi[k] - this.regularization * pu[k];
						double dqik = error * pu[k] - this.regularization * qi[k];

						pu[k] += this.learningRate * dpuk;
						qi[k] += this.learningRate * dqik;
					}

					this.setUserFactors(userIndex, pu);
					this.setItemFactors(itemIndex, qi);
				}

				for (int groupIndex = 0; groupIndex < GroupsManager.getInstance().getNumberOfGroups(); groupIndex++) {

					Group group = GroupsManager.getInstance().getGroupByIndex(groupIndex);

					if (group.hasRated(itemCode)) {

						Iterator<Double> ratings = group.getRatings(itemCode);
						double rating = this.merger.apply(ratings);

						double prediction = this.getGroupPrediction(groupIndex, itemIndex);

						double error = rating - prediction;

						// update factors

						double[] sg = this.getGroupFactors(groupIndex);
						double[] qi = this.getItemFactors(itemIndex);

						for (int k = 0; k < this.numFactors; k++) {
							double dsgk = error * qi[k] - this.regularization * sg[k];
							double dqik = error * sg[k] - this.regularization * qi[k];

							sg[k] += this.learningRate * dsgk;
							qi[k] += this.learningRate * dqik;
						}

						this.setGroupFactors(groupIndex, sg);
						this.setItemFactors(itemIndex, qi);
					}
				}
			}

			if (iter % 5 == 0) {
				double error = getPredictionError();
				System.out.println(" done! (error = " + error + ")");
			} else {
				System.out.println(" done!");
			}
		}
	}

	public double [] getUserFactors (int userIndex) {
		return this.usersFactors[userIndex];
	}

	private void setUserFactors (int userIndex, double [] factors) {
		this.usersFactors[userIndex] = factors;
	}

	public double [] getItemFactors (int itemIndex) {
		return this.itemsFactors[itemIndex];
	}

	private void setItemFactors (int itemIndex, double [] factors) {
		this.itemsFactors[itemIndex] = factors;
	}

	public double [] getGroupFactors (int groupIndex) {
		return this.groupsFactors[groupIndex];
	}

	public void setGroupFactors (int groupIndex, double [] factors) {
		this.groupsFactors[groupIndex] = factors;
	}

	public double getPrediction (int userIndex, int itemIndex) {
		double [] pu = this.getUserFactors(userIndex);
		double [] qi = this.getItemFactors(itemIndex);
		return Methods.dotProduct(pu, qi);
	}

	public double getGroupPrediction (int groupIndex, int itemIndex) {
		double [] sg = this.getGroupFactors(groupIndex);
		double [] qi = this.getItemFactors(itemIndex);
		return Methods.dotProduct(sg, qi);
	}

	private double random (double min, double max) {
		return Math.random() * (max - min) + min;
	}

	private double [] random (int size, double min, double max) {
		double [] d = new double [size];
		for (int i = 0; i < size; i++) d[i] = this.random(min, max);
		return d;
	}

	private double getPredictionError () {
		double error = 0;
		int count = 0;

		for (int itemIndex = 0; itemIndex < Kernel.getInstance().getNumberOfItems(); itemIndex++) {

			Item item = Kernel.getInstance().getItemByIndex(itemIndex);
			int itemCode = item.getItemCode();

			for (int groupIndex = 0; groupIndex < GroupsManager.getInstance().getNumberOfGroups(); groupIndex++) {
				Group group = GroupsManager.getInstance().getGroupByIndex(groupIndex);

				if (group.hasRated(itemCode)) {

					Iterator<Double> ratings = group.getRatings(itemCode);
					double rating = this.merger.apply(ratings);
					double prediction = this.getGroupPrediction(groupIndex, itemIndex);

					error += Math.abs(rating - prediction);
					count++;
				}
			}
		}

		return error / count;
	}
}

package main.java.gumf;

import cf4j.Item;
import cf4j.Kernel;
import cf4j.utils.Methods;
import main.java.data.GroupsManager;
import main.java.qualityMeasures.GroupQualityMeasures;

public class BiasedGrmf implements GroupRecommender {


	// model parameters

	private double [][] itemsFactors;
	private double [] itemsBias;

	private double [][] groupsFactors;
	private double [] groupsBias;


	// model hiper-paramenters

	private int numFactors;
	private int numIters;
	private double learningRate;
	private double regularization;


	public BiasedGrmf(int numFactors, int numIters, double regularization, double learningRate) {
		this.numFactors = numFactors;
		this.numIters = numIters;
		this.regularization = regularization;
		this.learningRate = learningRate;

		this.itemsFactors = new double [Kernel.getInstance().getNumberOfItems()][numFactors];
		this.itemsBias = new double [Kernel.getInstance().getNumberOfItems()];

		this.groupsFactors = new double [GroupsManager.getInstance().getNumberOfGroups()][numFactors];
		this.groupsBias = new double [GroupsManager.getInstance().getNumberOfGroups()];

		for (int itemIndex = 0; itemIndex < Kernel.getInstance().getNumberOfItems(); itemIndex++) {
			this.setItemFactors(itemIndex, this.random(this.numFactors, -1, 1));
			this.setItemBias(itemIndex, this.random(-1, 1));
		}

		for (int groupIndex = 0; groupIndex < GroupsManager.getInstance().getNumberOfGroups(); groupIndex++) {
			this.setGroupFactors(groupIndex, this.random(this.numFactors, -1, 1));
			this.setGroupBias(groupIndex, this.random(-1, 1));
		}
	}



	public void train () {

		System.out.println("\nProcessing BiasedGrmf...");

		for (int iter = 1; iter <= this.numIters; iter++) {

			System.out.print("iter " + iter + " of " + this.numIters + "...");

			for (int itemIndex = 0; itemIndex < Kernel.getInstance().getNumberOfItems(); itemIndex++) {

				Item item = Kernel.getInstance().getItemByIndex(itemIndex);

				for (int groupIndex = 0; groupIndex < GroupsManager.getInstance().getNumberOfGroups(); groupIndex++) {

					//double rating = GroupsManager.getInstance().getSumRating(groupIndex, itemIndex);
					//double rating = GroupsManager.getInstance().getGroupByIndex(groupIndex).getMinRating(itemIndex);
					double rating = 0; // deprecated
					if (rating == -1) continue; // group has not rated the item

					double prediction = this.getGroupPrediction(groupIndex, itemIndex);

					double error = rating - prediction;

					// update bias

					double bg = this.getGroupBias(groupIndex);
					double bi = this.getItemBias(itemIndex);

					bg += this.learningRate * (error - this.regularization * bg);
					bi += this.learningRate * (error - this.regularization * bi);

					this.setGroupBias(groupIndex, bg);
					this.setItemBias(itemIndex, bi);

					// update factors

					double [] sg = this.getGroupFactors(groupIndex);
					double [] qi = this.getItemFactors(itemIndex);

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

			System.out.print(" done!");

			double precision = GroupQualityMeasures.getPrecision(this,10,4,0.75);
			System.out.println(" (precision = " + precision + ")");
		}
	}

	public double getItemBias (int itemIndex) {
		return this.itemsBias[itemIndex];
	}

	public void setItemBias (int itemIndex, double bias) {
		this.itemsBias[itemIndex] = bias;
	}

	public double [] getItemFactors (int itemIndex) {
		return this.itemsFactors[itemIndex];
	}

	private void setItemFactors (int itemIndex, double [] factors) {
		this.itemsFactors[itemIndex] = factors;
	}

	public double getGroupBias (int groupIndex) {
		return this.groupsBias[groupIndex];
	}

	public void setGroupBias (int groupIndex, double bias) {
		this.groupsBias[groupIndex] = bias;
	}

	public double [] getGroupFactors (int groupIndex) {
		return this.groupsFactors[groupIndex];
	}

	public void setGroupFactors (int groupIndex, double [] factors) {
		this.groupsFactors[groupIndex] = factors;
	}

	public double getGroupPrediction (int groupIndex, int itemIndex) {
		//double average = GroupsManager.getInstance().getRatingAverage();
		double average = 0; // deprecated

		double [] sg = this.getGroupFactors(groupIndex);
		double [] qi = this.getItemFactors(itemIndex);

		double bg = this.getGroupBias(groupIndex);
		double bi = this.getItemBias(itemIndex);

		return average + bg + bi + Methods.dotProduct(sg, qi);
	}

	private double random (double min, double max) {
		return Math.random() * (max - min) + min;
	}

	private double [] random (int size, double min, double max) {
		double [] d = new double [size];
		for (int i = 0; i < size; i++) d[i] = this.random(min, max);
		return d;
	}
}

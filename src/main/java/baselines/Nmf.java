package main.java.baselines;

import cf4j.*;
import cf4j.model.matrixFactorization.FactorizationModel;
import cf4j.model.predictions.FactorizationPrediction;
import cf4j.qualityMeasures.MAE;
import cf4j.utils.Methods;

public class Nmf implements FactorizationModel {

	private final static String USER_FACTORS_KEY = "nmf-user-factors";
	private final static String ITEM_FACTORS_KEY = "nmf-item-factors";

	private int numFactors;
	private int numIters;


	public Nmf(int numFactors, int numIters) {

		this.numFactors = numFactors;
		this.numIters = numIters;

		for (int userIndex = 0; userIndex < Kernel.gi().getNumberOfUsers(); userIndex++) {
			this.setUserFactors(userIndex, this.random(this.numFactors, 0, 1));
		}

		for (int itemIndex = 0; itemIndex < Kernel.gi().getNumberOfItems(); itemIndex++) {
			this.setItemFactors(itemIndex, this.random(this.numFactors, 0, 1));
		}
	}


	public int getNumberOfTopics () {
		return this.numFactors;
	}

	public void train () {

		System.out.println("\nProcessing NMF...");

		for (int iter = 1; iter <= this.numIters; iter++) {

			Processor.getInstance().usersProcess(new UpdateUsersFactors(), false);
			Processor.getInstance().itemsProcess(new UpdateItemsFactors(), false);


//			Processor.getInstance().testUsersProcess(new FactorizationPrediction(this), false);
//			Processor.getInstance().testUsersProcess(new MAE(), false);
//			System.out.println("iter = " + iter + "; mae = " + Kernel.getInstance().getQualityMeasure("MAE"));

			if ((iter % 10) == 0) System.out.print(".");
			if ((iter % 100) == 0) System.out.println(iter + " iterations");
		}
	}


	public double [] getUserFactors (int userIndex) {
		User user = Kernel.gi().getUserByIndex(userIndex);
		return (double []) user.get(USER_FACTORS_KEY);
	}

	private void setUserFactors (int userIndex, double [] factors) {
		User user = Kernel.gi().getUserByIndex(userIndex);
		user.put(USER_FACTORS_KEY, factors);
	}

	public double [] getItemFactors (int itemIndex) {
		Item item = Kernel.gi().getItems()[itemIndex];
		return (double []) item.get(ITEM_FACTORS_KEY);
	}

	private void setItemFactors (int itemIndex, double [] factors) {
		Item item = Kernel.gi().getItemByIndex(itemIndex);
		item.put(ITEM_FACTORS_KEY, factors);
	}


	public double getPrediction (int userIndex, int itemIndex) {
		double [] wu = this.getUserFactors(userIndex);
		double [] hi = this.getItemFactors(itemIndex);
		return  Methods.dotProduct(wu, hi);
	}

	
	private class UpdateUsersFactors implements UsersPartible {

		@Override
		public void beforeRun() { }

		@Override
		public void run (int userIndex) {

			User user = Kernel.gi().getUsers()[userIndex];

			double [] wu = Nmf.this.getUserFactors(userIndex);

			double [] ratings = user.getRatings();
			double [] predictions = new double [user.getNumberOfRatings()];

			int [] indexes = new int [user.getNumberOfRatings()];

			int itemIndex = 0;

			for (int j = 0; j < user.getNumberOfRatings(); j++) {
				while (Kernel.gi().getItemByIndex(itemIndex).getItemCode() < user.getItemAt(j)) itemIndex++;
				predictions[j] = Nmf.this.getPrediction(userIndex, itemIndex);
				indexes[j] = itemIndex;
			}

			for (int k = 0; k < Nmf.this.numFactors; k++) {

				double sumRatings = 0;
				double sumPredictions = 0;

				for (int j = 0; j < user.getNumberOfRatings(); j++) {
					double [] hi = Nmf.this.getItemFactors(indexes[j]);
					sumRatings += hi[k] * ratings[j];
					sumPredictions += hi[k] * predictions[j];
				}

				wu[k] = wu[k] * sumRatings / (sumPredictions + 1E-10);
			}
		}

		@Override
		public void afterRun() { }
	}

	
	private class UpdateItemsFactors implements ItemsPartible {

		@Override
		public void beforeRun() { }

		@Override
		public void afterRun() { }

		@Override
		public void run(int itemIndex) {

			Item item = Kernel.gi().getItems()[itemIndex];

			double [] hi = Nmf.this.getItemFactors(itemIndex);

			double [] ratings = item.getRatings();
			double [] predictions = new double [item.getNumberOfRatings()];

			int [] indexes = new int [item.getNumberOfRatings()];

			int userIndex = 0;

			for (int v = 0; v < item.getNumberOfRatings(); v++) {
				while (Kernel.gi().getUserByIndex(userIndex).getUserCode() < item.getUserAt(v)) userIndex++;
				predictions[v] = Nmf.this.getPrediction(userIndex, itemIndex);
				indexes[v] = userIndex;
			}

			for (int k = 0; k < Nmf.this.numFactors; k++) {

				double sumRatings = 0;
				double sumPredictions = 0;

				for (int v = 0; v < item.getNumberOfRatings(); v++) {
					double [] wu = Nmf.this.getUserFactors(indexes[v]);
					sumRatings += wu[k] * ratings[v];
					sumPredictions += wu[k] * predictions[v];
				}

				hi[k] = hi[k] * sumRatings / (sumPredictions + 1E-10);
			}
		}
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

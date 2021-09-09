package main.java.baselines;

import java.util.Arrays;
import java.util.Iterator;

import cf4j.Item;
import main.java.data.Group;
import main.java.data.GroupsManager;
import main.java.gumf.GroupRecommender;
import main.java.misc.AggregationFunctions;
import org.ejml.simple.SimpleMatrix;

import cf4j.Kernel;
import cf4j.model.matrixFactorization.Pmf;
import cf4j.utils.Methods;

/**
 * Ortega, F., Hernando, A., Bobadilla, J., & Kang, J. H. (2016). Recommending items to group of users using matrix factorization based collaborative filtering. Information Sciences, 345, 313-324.
 */
public class Mfgu implements GroupRecommender {

	private int numFactors;
	private int numIters;
	private double learningRate;
	private double regularization;

	private Pmf pmf;

	private SimpleMatrix m;

	public Mfgu(int numFactors, int numIters, double regularization, double learningRate) {
		this.numFactors = numFactors;
		this.numIters = numIters;
		this.regularization = regularization;
		this.learningRate = learningRate;
	}

	public void train () {

		// train pmf model
		this.pmf = new Pmf(this.numFactors, this.numIters, this.regularization, this.learningRate, true);
		this.pmf.train();

		// prepare baselines model
		int rows = Kernel.getInstance().getNumberOfItems();
		int colls = this.pmf.getNumberOfTopics() + 1;

		SimpleMatrix A = new SimpleMatrix(rows, colls);

		A.setColumn(A.numCols() - 1, 0, 1);

		for (int i = 0; i < Kernel.getInstance().getNumberOfItems(); i++) {
			double [] factors = this.pmf.getItemFactors(i);

			for (int k = 0; k < this.pmf.getNumberOfTopics(); k++) {
				A.set(i, k, factors[k]);
			}
		}

		this.m = A.transpose()
				.mult(A)
				.plus(SimpleMatrix.identity(colls).scale(this.pmf.getLambda()))
				.invert()
				.mult(A.transpose());
	}

	public double getGroupPrediction (int groupIndex, int itemIndex) {

		double average = Kernel.getInstance().getRatingAverage();

		double [] groupFactors = this.getGroupFactors(groupIndex);

		double [] sg = Arrays.copyOf(groupFactors, this.pmf.getNumberOfTopics());
		double bg = groupFactors[groupFactors.length - 1];

		double [] qi = this.pmf.getItemFactors(itemIndex);
		double bi = this.pmf.getItemBias(itemIndex);

		return average + bg + bi + Methods.dotProduct(sg, qi);
	}

	public Pmf getFactorizationModel () {
		return this.pmf;
	}

	private double [] getGroupFactors (int groupIndex) {

		Group group = GroupsManager.getInstance().getGroupByIndex(groupIndex);

		int numItems = Kernel.getInstance().getNumberOfItems();

		SimpleMatrix s_gi = new SimpleMatrix (numItems, 1);
		s_gi.set(0);

		for (int itemIndex = 0; itemIndex < numItems; itemIndex++) {

			Item item = Kernel.getInstance().getItemByIndex(itemIndex);
			int itemCode = item.getItemCode();

			if (group.hasRated(itemCode)) {
				Iterator<Double> ratings = group.getRatings(itemCode);
				double rating = AggregationFunctions.min().apply(ratings); // least misery aggregation function

				double average = Kernel.getInstance().getRatingAverage();
				double itemBias = this.pmf.getItemBias(itemIndex);

				double s = rating - average - itemBias;

				s_gi.set(itemIndex, 0, s);
			}
		}

		SimpleMatrix aux = this.m.mult(s_gi);

		double [] factors = new double [this.pmf.getNumberOfTopics() + 1];
		for (int k = 0; k < aux.numRows(); k++) {
			factors[k] = aux.get(k, 0);
		}

		return factors;
	}
}

package com.github.knodis;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BiasedMF;
import es.upm.etsisi.cf4j.util.Range;
import es.upm.etsisi.cf4j.util.optimization.GridSearchCV;
import es.upm.etsisi.cf4j.util.optimization.ParamsGrid;

import java.io.IOException;

public class BiasedMFGridSearchCF {

    public static void main(String[] args) throws IOException {

        DataModel ml1m = BenchmarkDataModels.MovieLens1M();

        ParamsGrid paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("lambda", Range.ofDoubles(0.005, 0.005, 20));
        paramsGrid.addParam("gamma", Range.ofDoubles(0.005, 0.005, 20));
        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});

        paramsGrid.addFixedParam("seed", 42L);

        GridSearchCV gridSearchCV = new GridSearchCV(ml1m, paramsGrid, BiasedMF.class, MAE.class, 5);
        gridSearchCV.fit();

        gridSearchCV.printResults(10);
    }
}

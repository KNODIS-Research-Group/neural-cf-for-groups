package main.java.misc;

import java.util.Iterator;
import java.util.function.Function;

public class AggregationFunctions {

    public static Function <Iterator <Double>, Double> min () {
        return (Iterator <Double> iter) -> {
          double min = Double.MAX_VALUE;
          while (iter.hasNext()) min = Math.min(min, iter.next());
          return min;
        };
    }

    public static Function <Iterator <Double>, Double> max () {
        return (Iterator <Double> iter) -> {
            double max = Double.MIN_VALUE;
            while (iter.hasNext()) max = Math.max(max, iter.next());
            return max;
        };
    }

    public static Function <Iterator <Double>, Double> sum () {
        return (Iterator <Double> iter) -> {
            double sum = 0;
            while (iter.hasNext()) sum += iter.next();
            return sum;
        };
    }

    public static Function <Iterator <Double>, Double> prod () {
        return (Iterator <Double> iter) -> {
            double prod = 1;
            while (iter.hasNext()) prod *= iter.next();
            return prod;
        };
    }

    public static Function <Iterator <Double>, Double> avg () {
        return (Iterator <Double> iter) -> {
            double sum = 0;
            int count = 0;
            while (iter.hasNext()) {
                sum += iter.next();
                count++;
            }
            return sum / count;
        };
    }

    public static Function <Iterator <Double>, Double> geo () {
        return (Iterator <Double> iter) -> {
            double prod = 1;
            int count = 0;
            while (iter.hasNext()) {
                prod *= iter.next();
                count++;
            }
            return Math.pow(prod, 1.0 / count);
        };
    }
}

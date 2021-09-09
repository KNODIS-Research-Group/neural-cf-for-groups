package main.java.experiments;

import cf4j.Kernel;

public class DatasetProperties {

    final static String CF4J_FILE = "Flixter.cf4j";

    public static void main (String [] args) {
        Kernel.getInstance().readKernel(CF4J_FILE);
        System.out.println(Kernel.getInstance().getKernelInfo());
    }
}

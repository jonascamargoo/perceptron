public class RNAMain {
    public static double[][] base = {
        {{0, 0}, {0}},
        {{0, 1}, {0}},
        {{1, 0}, {0}},
        {{1, 1}, {1}}
    }

    public static void main(String[] args) {
        Perceptron p = new Perceptron(2, 1, 0.3);
        for (int e = 0; e < 10000; e++) {
            double erroE = 0;
            for(a = 0; a < base.length; a++) {
                double[]x = base[a][0];
                double[]y = base[a][1];
                double out = rna.treinar
            }
        }

     
    }
}
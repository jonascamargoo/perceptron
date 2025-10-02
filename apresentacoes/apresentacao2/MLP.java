package apresentacoes.apresentacao2;

import java.util.Random;

public class MLP extends RNA {

    private int qtd_in;
    private int qtd_h; // Quantidade de neuronios na camada intermediaria
    private int qtd_out;
    private double[][] wh, wo;
    private double ni = 0.3; // coef de aprendizado

    public MLP(int qtd_in, int qtd_h, int qtd_out, double ni) {
        this(qtd_in, qtd_h, qtd_out);
        this.ni = ni;
    }

    public MLP(int qtd_in, int qtd_h, int qtd_out) {
        this.qtd_in = qtd_in;
        this.qtd_out = qtd_out;
        this.qtd_h = qtd_h;
        this.wh = new double[qtd_in + 1][qtd_h];
        this.wo = new double[qtd_h + 1][qtd_out];

        Random aleatorio = new Random();
        for (int i = 0; i < wh.length; i++) {
            for (int j = 0; j < wh[0].length; j++) {
                wh[i][j] = aleatorio.nextDouble() * 0.6 - 0.3;
            }
        }
        for (int i = 0; i < wo.length; i++) {
            for (int j = 0; j < wo[0].length; j++) {
                wo[i][j] = aleatorio.nextDouble() * 0.6 - 0.3;
            }
        }

    }

    @Override
    public double[] treinar(double[] x_in, double[] y) {
        
        double[] x = new double[x_in.length + 1];
        x[x.length - 1] = 1; // termo de bias
        for (int i = 0; i < x_in.length; i++) {
            x[i] = x_in[i];
        }

        // Obtem as saidas dos neuronios da camada intermediaria
        double[] H = new double[qtd_h + 1];
        for (int h = 0; h < H.length - 1; h++) {
            double u = 0;
            for (int i = 0; i < x.length; i++) { 
                u += x[i] * this.wh[i][h];
            }
            H[h] = 1 / (1 + Math.exp(-u));
        }
        H[H.length - 1] = 1; // termo de bias

        double[] out = new double[qtd_out];
        for(int j = 0; j < out.length; j++) {
            double u = 0;
            for(int h = 0; h < H.length; h++) {
                u = u + H[h] * this.wo[h][j];
            }
            out[j] = 1/(1 + Math.exp(-u));
        }

        //==========================================
        // calculo dos deltas
        //==========================================
        double[] DO = new double[qtd_out];
        for(int j = 0; j < DO.length; j++) {
            DO[j] = out[j] * (1 - out[j]) * (y[j] - out[j]);
        }

        double[] DH = new double[qtd_h + 1];
        for(int h = 0; h < DH.length; h++) {
            double s = 0;
            for(int j = 0; j < out.length; j++) {
                s += DO[j] * this.wo[h][j];
            }
            DH[h] = H[h] * (1 - H[h]) * s;
        }


        //==========================================
        // Ajuste dos pesos
        //==========================================
        for( int i = 0; i < wh.length; i++) {
            for( int h = 0; h < wh[0].length; h++) {
                wh[i][h] += ni * DH[h] * x[i];
            }
        }

        for( int h = 0; h < wo.length; h++) {
            for( int j = 0; j < wo[0].length; j++) {
                wo[h][j] += ni * DO[j] * H[h];
            }
        }


        return out;

    }

    @Override
    public double[] executar(double[] x_in) {
        double[] x = new double[x_in.length + 1];
        x[x.length - 1] = 1; // termo de bias
        for (int i = 0; i < x_in.length; i++) {
            x[i] = x_in[i];
        }

        // Obtem as saidas dos neuronios da camada intermediaria
        double[] H = new double[qtd_h + 1];
        for (int h = 0; h < H.length - 1; h++) {
            double u = 0;
            for (int i = 0; i < x.length; i++) { 
                u += x[i] * this.wh[i][h];
            }
            H[h] = 1 / (1 + Math.exp(-u));
        }
        H[H.length - 1] = 1; // termo de bias

        double[] out = new double[qtd_out];
        for(int j = 0; j < out.length; j++) {
            double u = 0;
            for(int h = 0; h < H.length; h++) {
                u = u + H[h] * this.wo[h][j];
            }
            out[j] = 1/(1 + Math.exp(-u));
        }
        return out;
    }

    // Implementação da MLP aqui

}

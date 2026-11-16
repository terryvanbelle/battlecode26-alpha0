package zdp;

import java.io.*;
import java.util.*;

/** T0 parity harness: each input line "p T floor mask(0/1 chars) z0 z1 ..." -> "k pbits fw0bits fw1bits ..." (fw: DNet.floorWeights,
 *  "null" when the point has no groups). Runs the bot's own DNet.sample with D.RL on. */
public class T0 {
    public static void main(String[] a) throws IOException {
        BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter w = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
        for (String line; (line = r.readLine()) != null; ) {
            String[] t = line.trim().split(" ");
            int p = Integer.parseInt(t[0]);
            float T = Float.parseFloat(t[1]), floor = Float.parseFloat(t[2]);
            String ms = t[3];
            int n = ms.length();
            boolean[] m = new boolean[n];
            for (int k = 0; k < n; k++) m[k] = ms.charAt(k) == '1';
            float[] z = new float[n];
            for (int k = 0; k < n; k++) z[k] = Float.intBitsToFloat((int) Long.parseLong(t[4 + k]));
            float[] fw = DNet.floorWeights(p, m, n);
            int k = DNet.sample(z, m, T, floor, false, fw);
            StringBuilder b = new StringBuilder();
            b.append(DNet.lastK).append(' ').append(Float.floatToIntBits(DNet.lastP));
            if (fw == null) b.append(" null");
            else for (int j = 0; j < n; j++) b.append(' ').append(Float.floatToIntBits(fw[j]));
            w.println(b);
        }
        w.flush();
    }
}

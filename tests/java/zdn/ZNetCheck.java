package zdn;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Test harness for tests/test_zdn_java.py: loads a packed zdn net (zdn_pack.py's chars, written as big-endian uint16)
 * with ZNet.java and prints the logits of each case.
 *
 *   java zdn.ZNetCheck NET CASES
 *   CASES: one case per line: head nIn x[nIn] nCtx ctx[nCtx] n D (kind x[D]) * n      (head -1: step scorer, r: register r)
 *   output: one line per case, the n logits ("-inf" for unknown kinds)
 */
public final class ZNetCheck {
    public static void main(String[] args) throws IOException {
        byte[] b = Files.readAllBytes(Paths.get(args[0]));
        char[] c = new char[b.length / 2];
        for (int i = 0; i < c.length; i++) c[i] = (char) ((b[2 * i] & 0xFF) << 8 | (b[2 * i + 1] & 0xFF));
        ZNet net = ZNet.of(new String[] {new String(c)});
        StringBuilder out = new StringBuilder();
        for (String line : Files.readAllLines(Paths.get(args[1]))) {
            if (line.isBlank()) continue;
            StringTokenizer t = new StringTokenizer(line);
            int head = Integer.parseInt(t.nextToken());
            float[] x = new float[Integer.parseInt(t.nextToken())];
            for (int i = 0; i < x.length; i++) x[i] = Float.parseFloat(t.nextToken());
            float[] ctx = new float[Integer.parseInt(t.nextToken())];
            for (int i = 0; i < ctx.length; i++) ctx[i] = Float.parseFloat(t.nextToken());
            int n = Integer.parseInt(t.nextToken()), D = Integer.parseInt(t.nextToken());
            int[] kind = new int[n];
            int[][] xs = new int[n][D];
            for (int i = 0; i < n; i++) {
                kind[i] = Integer.parseInt(t.nextToken());
                for (int d = 0; d < D; d++) xs[i][d] = Integer.parseInt(t.nextToken());
            }
            float[] h = net.trunk(x);
            ZNet.Scorer sc = head < 0 ? net.step : net.regs[head];
            float[] z = ZNet.logits(sc, h, ctx, n, kind, xs);
            for (int i = 0; i < n; i++) out.append(i == 0 ? "" : " ").append(z[i] == Float.NEGATIVE_INFINITY ? "-inf" : Float.toString(z[i]));
            out.append('\n');
        }
        System.out.print(out);
    }
}

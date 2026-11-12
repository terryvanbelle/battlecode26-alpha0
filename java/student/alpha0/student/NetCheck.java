package alpha0.student;

import java.io.*;
import java.util.zip.GZIPInputStream;

/** Prints "argmax value" for the first K records of a file using the Java Net (parity test vs PyTorch). */
public final class NetCheck {
    public static void main(String[] a) throws Exception {
        Net net = new Net(new File(a[0]));
        int k = Integer.parseInt(a[2]);
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(a[1]))))) {
            byte[] head = new byte[Records.HEADER], obs = new byte[Obs.SIZE], mb = new byte[Records.MASK_BYTES];
            boolean[] mask = new boolean[Actions.N];
            for (int r = 0; r < k; r++) {
                in.readFully(head); in.readFully(obs); in.readFully(mb);
                for (int i = 0; i < Actions.N; i++) mask[i] = (mb[i >> 3] >> (i & 7) & 1) != 0;
                float v = net.forward(obs);
                System.out.println(net.act(mask, 0, null) + " " + v);
            }
        }
    }
}

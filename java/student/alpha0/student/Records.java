package alpha0.student;

import java.io.*;
import java.util.zip.GZIPOutputStream;

/**
 * Fixed-size binary decision records (student-records-v1), gzip-compressed, big-endian.
 * header 16 bytes: int32 robotId, int32 round, int16 label, int8 team, int8 isKing,
 * int8 step, int8 flags (bit0: label was not in our legal mask), int16 reserved;
 * then Obs.SIZE uint8 observation bytes, then MASK_BYTES of legal-mask bits (LSB-first).
 */
public final class Records implements Closeable {
    public static final int HEADER = 16, MASK_BYTES = (Actions.N + 63) / 64 * 8;
    public static final int SIZE = HEADER + Obs.SIZE + MASK_BYTES;
    private final DataOutputStream out;
    private final byte[] maskBuf = new byte[MASK_BYTES];
    public long count;

    public Records(File file) throws IOException {
        out = new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(file), 1 << 16), 1 << 16));
    }

    public synchronized void write(int robotId, int round, int label, int team, boolean king, int step, boolean labelNotLegal,
                                   byte[] obs, boolean[] mask) throws IOException {
        out.writeInt(robotId);
        out.writeInt(round);
        out.writeShort(label);
        out.writeByte(team);
        out.writeByte(king ? 1 : 0);
        out.writeByte(Math.min(step, 127));
        out.writeByte(labelNotLegal ? 1 : 0);
        out.writeShort(0);
        out.write(obs, 0, Obs.SIZE);
        java.util.Arrays.fill(maskBuf, (byte) 0);
        for (int i = 0; i < Actions.N; i++) if (mask[i]) maskBuf[i >> 3] |= (byte) (1 << (i & 7));
        out.write(maskBuf);
        count++;
    }

    @Override public synchronized void close() throws IOException { out.close(); }
}

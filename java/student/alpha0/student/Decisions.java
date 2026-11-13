package alpha0.student;

import battlecode.common.*;
import java.io.*;
import java.util.zip.GZIPOutputStream;

/**
 * Decision-record channel (decisions-v1). A bot calls rc.setIndicatorString(s) with s.charAt(0) == '\uDEC0';
 * the hooked RobotControllerImpl hands s to take() instead of the indicator string.
 * Wire format of s (chars as unsigned 16-bit): [0]=0xDEC0, [1]=point, [2]=label, [3]=n, [4..4+n)=features.
 * File record (gzip, big-endian), 14 + 2n bytes: int32 robotId, int16 round, int8 team (0 A / 1 B), int8 isKing,
 * int16 point, int16 label, int16 n, then n int16 features (all 16-bit fields hold the char value; read as uint16).
 * Robots run one at a time, so static state is safe.
 */
public final class Decisions {
    static final DataOutputStream[] sinks = new DataOutputStream[2];
    public static final long[] count = new long[2];
    public static long errors;

    public static void open(int team, File file) throws IOException {
        sinks[team] = new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(file), 1 << 16), 1 << 16));
    }

    public static void closeAll() {
        for (int t = 0; t < 2; t++) {
            if (sinks[t] == null) continue;
            try { sinks[t].close(); } catch (IOException e) { errors++; }
            sinks[t] = null;
        }
    }

    public static void take(RobotController rc, String s) {
        try {
            int team = rc.getTeam() == Team.A ? 0 : rc.getTeam() == Team.B ? 1 : -1;
            if (team < 0) { errors++; return; }
            DataOutputStream out = sinks[team];
            if (out == null) return;
            if (s.length() < 4 || s.length() != 4 + s.charAt(3)) { errors++; return; }
            int n = s.charAt(3);
            out.writeInt(rc.getID());
            out.writeShort(rc.getRoundNum());
            out.writeByte(team);
            out.writeByte(rc.getType() == UnitType.RAT_KING ? 1 : 0);
            out.writeShort(s.charAt(1));
            out.writeShort(s.charAt(2));
            out.writeShort(n);
            for (int i = 0; i < n; i++) out.writeShort(s.charAt(4 + i));
            count[team]++;
        } catch (Throwable e) { errors++; }
    }
}

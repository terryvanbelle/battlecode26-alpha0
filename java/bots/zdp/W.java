package zdp;

/** Decision-net weights, one packed string per decision point (see DNet). Net-mode builds replace this file
 *  (scripts/build_dpnet.py); the recording build has none. A: per-point count of reachable options (missing or 0:
 *  all of the net's outputs); FL: per-point sampling floor (missing or < 0: D.NET_FLOOR). */
final class W {
    static final String[] P = new String[0];
    static final short[] A = {};
    static final float[] FL = {};

    private W() {}
}

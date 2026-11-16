package zdp;

/** Decision-net weights, one packed string per decision point (see DNet). Net-mode builds replace this file
 *  (scripts/build_dpnet.py); the recording build has none. A: per-point count of reachable options (missing or 0:
 *  all of the net's outputs); FL: per-point sampling floor (missing or < 0: D.NET_FLOOR); G, GW: per-point floor groups
 *  (DNet.floorWeights; missing or empty: the floor is uniform over the allowed options). */
final class W {
    static final String[] P = new String[0];
    static final short[] A = {};
    static final float[] FL = {};
    static final String[] G = {};
    static final float[][] GW = {};

    private W() {}
}

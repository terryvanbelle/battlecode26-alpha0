package zdp;

/** Indicator-string telemetry. Tags are stable so replay tools can grep them. */
public final class Debug {
    public static final boolean ON = false;
    /** Bytecode probe: search pops and work() bytecodes, and the indicator's own cost, in the indicator (fx=). */
    public static final boolean PROBE = false;
    public static int indBc;

    /** Overrun locator: mark(k) at turn stages; the first stage reached after the round advanced is kept. */
    public static int stage, ovStage = -1, startRound;
    public static boolean crossed;

    public static int lastBc, hotStage, hotBc;   // the costliest stage segment this turn

    public static void begin(int round) { startRound = round; stage = 0; crossed = false; lastBc = 0; hotBc = 0; }

    public static void mark(int k) {
        // ledger (z2h): u6x's count at this point; the hot-stage branch below costs bytecodes, so it must branch as u6x's
        int bc = battlecode.common.Clock.getBytecodeNum() - (G.debt += 8) + 6;
        if (!crossed && G.rc.getRoundNum() != startRound) { crossed = true; ovStage = stage * 100 + k; }
        else if (bc - lastBc > hotBc) { hotBc = bc - lastBc; hotStage = stage * 100 + k; }
        lastBc = bc;
        stage = k;
    }

    public static void indicator() {
        StringBuilder sb = new StringBuilder(96);
        if (G.king) {
            sb.append("K sp=").append(King.spawned).append(" tr=").append(King.traps).append(" inc=").append(Econ.rate100()).append(" sur=").append(Econ.surplus()).append(" km=").append(King.mineWhy).append(Comms.numMines).append(" ml=");
            // the team mine list, two characters a mine ('0' + x, '0' + y)
            for (int k = 0; k < Comms.numMines; k++) sb.append((char) ('0' + Comms.mines[k].x)).append((char) ('0' + Comms.mines[k].y));
            sb.append(" hm=").append(King.homeMine == null ? "-" : King.homeMine.x + "," + King.homeMine.y).append(" kd=").append(King.digs).append(" sw=").append(King.spawnWhy).append(King.sawEnemy ? "E" : "");
        } else {
            sb.append(Baby.role == Baby.FIGHT ? "F" : Baby.role == Baby.ATTACK ? "A" : "G")
              .append(" dl=").append(Baby.delivered).append(" mm=").append(Baby.myMineCount)
              .append(" bc=").append(Baby.bitesCat)
              .append(" br=").append(Baby.bitesRat)
              .append(" tr=").append(Baby.trapsPlaced)
              .append(" rt=").append(Baby.ratTraps).append(" nap=").append(Baby.naps).append(" th=").append(Baby.throwsDone).append(" gl=").append(Baby.glances)
              .append(" dg=").append(Nav.digs).append(" fl=").append(Flow.builds).append(":").append(Flow.done).append(":").append(Flow.lastTurns).append(":").append(Flow.steps).append(" mv=").append(Baby.why);
            if (Baby.mineTarget != null) sb.append(" mt=").append(Baby.mineTarget.x).append(',').append(Baby.mineTarget.y);
            // navigation state: nav target and stall, flow target (b building, r ready, - stale) and robot-held turns
            if (Nav.target != null) sb.append(" nt=").append(Nav.target.x).append(',').append(Nav.target.y).append('/').append(Nav.stall);
            if (Flow.tgt != null) sb.append(" ft=").append(Flow.tgt.x).append(',').append(Flow.tgt.y)
              .append(Flow.building ? 'b' : Flow.builtRound >= 0 ? 'r' : '-').append(Flow.held).append('/').append(Flow.plen - Flow.ppos);
            if (Cfg.BFLOW && BitFlow.tgt != null) sb.append(" bf=").append(BitFlow.tgt.x).append(',').append(BitFlow.tgt.y)
              .append(BitFlow.building ? 'b' : '-').append(BitFlow.wave).append(':').append(BitFlow.meD).append(':').append(BitFlow.builds)
              .append(':').append(BitFlow.steps).append(':').append(BitFlow.fails).append(':').append(BitFlow.restarts).append(':').append(BitFlow.held);
            int[] fe = Flow.ends;
            sb.append(" fe=").append(fe[0]).append(':').append(fe[1]).append(':').append(fe[2]).append(':').append(fe[3]).append(':').append(fe[4]).append(':').append(fe[5]).append(':').append(Flow.fails).append(':').append(Flow.hits);
            if (PROBE) sb.append(" fx=").append(Flow.pops).append(':').append(Flow.workBc).append(':').append(indBc);
            if (Cfg.CASH_BANK > 0 && Baby.cashSite != null)
                sb.append(" cs=").append(Baby.cashSite.x).append(',').append(Baby.cashSite.y).append(Baby.cashLead ? 'L' : 'J').append(Baby.cashStart);
        }
        if (Baby.raidTarget != null) sb.append(" rz=").append(Baby.raidTarget.x).append(',').append(Baby.raidTarget.y);
        sb.append(" sym=").append(Sym.mask);
        if (Sym.contradictions > 0) sb.append(" SYMX=").append(Sym.contradictions);
        if (!G.king && Baby.lowBc > 0) sb.append(" lb=").append(Baby.lowBc);
        if (Cfg.FACE_HOLD) sb.append(" fh=").append(Baby.faceHolds);
        if (Cfg.BURN) sb.append(" burn=").append(G.king ? (King.burnHeard ? 1 : 0) : Baby.burnRound).append(Baby.adaptRisk ? "*" : "");
        if (Cfg.WOUNDED_HP > 0 && !G.king && Baby.wounded > 0) sb.append(" wd=").append(Baby.wounded);
        if (!G.king && Baby.defends > 0) sb.append(" df=").append(Baby.defends);
        if (!G.king && Baby.rescues > 0) sb.append(" rs=").append(Baby.rescues);
        if (Cfg.GUARD_CARRY_R > 0 && !G.king && Baby.carryGuards > 0) sb.append(" cg=").append(Baby.carryGuards);
        if (!G.king && Baby.presses > 0) sb.append(" pk=").append(Baby.presses);
        if (!G.king && Baby.guards > 0) sb.append(" gd=").append(Baby.guards);
        if (RobotPlayer.overruns > 0) sb.append(" OVR=").append(RobotPlayer.overruns).append(" OVS=").append(ovStage);
        if (RobotPlayer.nearMiss > 0) sb.append(" NEAR=").append(RobotPlayer.nearMiss);
        if (RobotPlayer.exceptions > 0) sb.append(" EXC=").append(RobotPlayer.exceptions).append(':').append(lastExc);
        sb.append(" hot=").append(hotStage).append(':').append(hotBc);
        sb.append(" mbc=").append(RobotPlayer.maxBc);
        G.rc.setIndicatorString(sb.toString());
    }

    /** Replay channel: replays keep each turn's bytecode count but drop indicator strings, so tag() pads the turn's
     *  end until the count mod 128 is a 7-bit tag. Bits 0-4: a baby's move decision (index in WHY) or a king's spawn
     *  decision (index in KWHY); bit 5: role FIGHT (GATHER and ATTACK both 0) / a king with enemies in view;
     *  bit 6: even parity of bits 0-5.
     *  tools/smuggle.py decodes it; keep the alphabets in step with it. */
    public static final boolean TAG = true;
    public static final String WHY = "-CDEHKMPSWacdfghkmprswyzxTRAe", KWHY = "-nsfcbd";
    /** Bytecodes the engine counts after tag()'s getBytecodeNum() read, the pad aside (calibrated on a local game:
     *  11 in tag() plus its return). Recalibrate if the engine jar or javac changes (smuggle.py: decode rate < 98%). */
    static final int TAG_K = 12;
    /** No pad with fewer bytecodes left than this (the pad costs at most 127). */
    static final int TAG_MIN_LEFT = 400;

    static int tagValue() {
        int w, hi;
        if (G.king) { w = KWHY.indexOf(King.spawnWhy); hi = King.sawEnemy ? 32 : 0; }
        else { w = WHY.indexOf(Baby.why); hi = Baby.role == Baby.FIGHT ? 32 : 0; }
        int v = (w < 0 ? 31 : w) | hi;
        return Integer.bitCount(v) % 2 == 0 ? v : v | 64;
    }

    /** Pad this turn so that the engine's final bytecode count is tagValue() mod 128. Call right before Clock.yield(). */
    public static void tag() {
        int v = tagValue();
        if (battlecode.common.Clock.getBytecodesLeft() < TAG_MIN_LEFT) return;
        int p = (v - battlecode.common.Clock.getBytecodeNum() - TAG_K) & 127;
        int n = 0;
        switch (p) {
            case 127: n++; case 126: n++; case 125: n++; case 124: n++; case 123: n++; case 122: n++; case 121: n++; case 120: n++;
            case 119: n++; case 118: n++; case 117: n++; case 116: n++; case 115: n++; case 114: n++; case 113: n++; case 112: n++;
            case 111: n++; case 110: n++; case 109: n++; case 108: n++; case 107: n++; case 106: n++; case 105: n++; case 104: n++;
            case 103: n++; case 102: n++; case 101: n++; case 100: n++; case 99: n++; case 98: n++; case 97: n++; case 96: n++;
            case 95: n++; case 94: n++; case 93: n++; case 92: n++; case 91: n++; case 90: n++; case 89: n++; case 88: n++;
            case 87: n++; case 86: n++; case 85: n++; case 84: n++; case 83: n++; case 82: n++; case 81: n++; case 80: n++;
            case 79: n++; case 78: n++; case 77: n++; case 76: n++; case 75: n++; case 74: n++; case 73: n++; case 72: n++;
            case 71: n++; case 70: n++; case 69: n++; case 68: n++; case 67: n++; case 66: n++; case 65: n++; case 64: n++;
            case 63: n++; case 62: n++; case 61: n++; case 60: n++; case 59: n++; case 58: n++; case 57: n++; case 56: n++;
            case 55: n++; case 54: n++; case 53: n++; case 52: n++; case 51: n++; case 50: n++; case 49: n++; case 48: n++;
            case 47: n++; case 46: n++; case 45: n++; case 44: n++; case 43: n++; case 42: n++; case 41: n++; case 40: n++;
            case 39: n++; case 38: n++; case 37: n++; case 36: n++; case 35: n++; case 34: n++; case 33: n++; case 32: n++;
            case 31: n++; case 30: n++; case 29: n++; case 28: n++; case 27: n++; case 26: n++; case 25: n++; case 24: n++;
            case 23: n++; case 22: n++; case 21: n++; case 20: n++; case 19: n++; case 18: n++; case 17: n++; case 16: n++;
            case 15: n++; case 14: n++; case 13: n++; case 12: n++; case 11: n++; case 10: n++; case 9: n++; case 8: n++;
            case 7: n++; case 6: n++; case 5: n++; case 4: n++; case 3: n++; case 2: n++; case 1: n++;
            default:
        }
    }

    static String lastExc = "";

    /** Remember the latest exception as type@stage (the last mark() reached; the engine sandbox forbids stack
     *  traces): A array index, N null, Z arithmetic, G game action, C cast, X other. Shown after EXC=. */
    static void exception(Exception e) {
        char t = e instanceof ArrayIndexOutOfBoundsException ? 'A' : e instanceof NullPointerException ? 'N'
            : e instanceof ArithmeticException ? 'Z' : e instanceof battlecode.common.GameActionException ? 'G'
            : e instanceof ClassCastException ? 'C' : 'X';
        lastExc = t + "@" + stage;
    }
}

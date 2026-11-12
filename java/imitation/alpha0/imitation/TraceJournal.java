package alpha0.imitation;

import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;

/** Exclusive bounded transport. Sealing requires an external whole-world terminal proof. */
public final class TraceJournal implements AutoCloseable {
    private static final int MAX_RECORD=1024*1024;
    private final OutputStream stream;
    private final String game;
    private final long limit;
    private long bytes,sequence;
    private String previous="0".repeat(64),lastKind;
    private boolean closed,failed,sealed;
    private final Map<String,long[]> turns=new HashMap<>();
    public TraceJournal(Path path,String gameId,long maxBytes) throws IOException {
        hash(gameId);if(maxBytes<=0)throw new IllegalArgumentException("invalid journal cap");
        game=gameId;limit=maxBytes;
        stream=Files.newOutputStream(path,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE);
    }
    public static void hash(String value) {
        if(value==null || !value.matches("[0-9a-f]{64}"))throw new IllegalArgumentException("invalid hash");
    }
    /** Restricted JSON tree: no floats, arbitrary objects, or private object serialization. */
    private static void tree(Object value) {
        if(value==null || value instanceof String || value instanceof Boolean || value instanceof Integer || value instanceof Long)return;
        if(value instanceof Map<?,?> m){for(var e:m.entrySet()){if(!(e.getKey() instanceof String key) || !key.chars().allMatch(c->c>=32 && c<=126))throw new IllegalArgumentException("non-string key");tree(e.getValue());}return;}
        if(value instanceof List<?> l){for(Object x:l)tree(x);return;}
        throw new IllegalArgumentException("unsupported JSON value");
    }
    private static String quote(String value) {
        StringBuilder out=new StringBuilder("\"");
        for(char c:value.toCharArray()) {
            switch(c) {
                case '"': out.append("\\\"");break;
                case '\\': out.append("\\\\");break;
                case '\b': out.append("\\b");break;
                case '\f': out.append("\\f");break;
                case '\n': out.append("\\n");break;
                case '\r': out.append("\\r");break;
                case '\t': out.append("\\t");break;
                default: if(c<32 || c>126)out.append(String.format(java.util.Locale.ROOT,"\\u%04x",(int)c));else out.append(c);
            }
        }
        return out.append('"').toString();
    }
    private static String encode(Object value) {
        if(value==null)return "null";
        if(value instanceof String s)return quote(s);
        if(value instanceof Map<?,?> m) {
            TreeMap<String,Object> sorted=new TreeMap<>();m.forEach((k,v)->sorted.put((String)k,v));
            ArrayList<String> fields=new ArrayList<>();sorted.forEach((k,v)->fields.add(quote(k)+":"+encode(v)));
            return "{"+String.join(",",fields)+"}";
        }
        if(value instanceof List<?> l){ArrayList<String> fields=new ArrayList<>();for(Object x:l)fields.add(encode(x));return "["+String.join(",",fields)+"]";}
        return value.toString();
    }
    public static byte[] canonical(Object value) throws IOException {tree(value);return encode(value).getBytes(java.nio.charset.StandardCharsets.US_ASCII);}
    public static String digest(Object value) throws IOException {
        try{return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(canonical(value)));}
        catch(NoSuchAlgorithmException e){throw new AssertionError(e);}
    }
    public synchronized void append(int robot,String team,int round,int turn,String kind,Map<String,Object> payload) throws IOException {
        try {
            if(closed || failed || sealed)throw new IllegalStateException("journal unavailable");
            if(robot<0 || round<0 || turn<0 || !(team.equals("A")||team.equals("B")))throw new IllegalArgumentException("invalid identity");
            if(!(kind.equals("action")||kind.equals("turn_end")))throw new IllegalArgumentException("invalid kind");
            String key=team+":"+robot;long[] prior=turns.get(key);
            if(prior!=null && (round<prior[0] || (round==prior[0] && turn<prior[1])))throw new IllegalArgumentException("backward history");
            Map<String,Object> row=new TreeMap<>();
            row.put("schema","public-controller-trace-v1");row.put("game_id",game);row.put("sequence",sequence);
            row.put("robot_id",robot);row.put("team",team);row.put("round",round);row.put("turn",turn);
            row.put("kind",kind);row.put("payload",payload);row.put("previous_sha256",previous);
            String next=digest(row);row.put("sha256",next);byte[] encoded=canonical(row);
            if(encoded.length+1>MAX_RECORD || encoded.length+1>limit-bytes)throw new IOException("journal capacity exceeded");
            stream.write(encoded);stream.write('\n');stream.flush();
            bytes+=encoded.length+1;sequence++;previous=next;lastKind=kind;turns.put(key,new long[]{round,turn});
        } catch(IOException|RuntimeException e){failed=true;throw e;}
    }
    /** Caller must independently prove terminal completion; closing alone never seals. */
    public synchronized void seal(Path path,String terminal,String runtime,String teacher,String source) throws IOException {
        if(closed || failed || sealed || sequence==0 || !"turn_end".equals(lastKind))throw new IllegalStateException("incomplete journal");
        RobotThreadRecorder.requireHealthy();
        hash(terminal);hash(runtime);hash(teacher);hash(source);
        Map<String,Object> seal=new TreeMap<>();seal.put("game_id",game);seal.put("complete",true);
        seal.put("record_count",sequence);seal.put("last_sha256",previous);seal.put("terminal_sha256",terminal);
        seal.put("runtime_identity",runtime);seal.put("teacher_identity",teacher);seal.put("source_spec_sha256",source);
        Files.write(path,canonical(seal),StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE);sealed=true;
    }
    public synchronized void invalidate(){failed=true;}
    public synchronized void close() throws IOException {closed=true;stream.close();}
}
